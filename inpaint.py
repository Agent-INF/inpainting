from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import thread
import tensorflow as tf
import numpy as np
from utils import save
from utils import load
from utils import read_by_batch
from utils import preprocess_image
from utils import save_images
from utils import compute_psnr_ssim

FLAG = tf.app.flags
FLAGS = FLAG.FLAGS
FLAG.DEFINE_string('dataname', 'inpaint', 'The dataset name')
FLAG.DEFINE_string('mode', 'train', 'Chose mode from (train, validate, test)')
FLAG.DEFINE_string('block', 'random', 'Block location, chose from (center, random)')
FLAG.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')
FLAG.DEFINE_float('lamb_rec', 0.999, 'Weight for reconstruct loss.')
FLAG.DEFINE_float('lamb_adv', 0.001, 'Weight for adversarial loss.')
FLAG.DEFINE_float('lamb_tv', 1e-6, 'Weight for TV loss.')
FLAG.DEFINE_boolean('use_l1', False,
                    'If True, use L1 distance for rec_loss, else use L2 distance')
FLAG.DEFINE_boolean('weight_clip', False,
                    'When updating G & D, clip weights or not.')
FLAG.DEFINE_integer('image_size', 128, 'Image size.')
FLAG.DEFINE_integer('image_channel', 1,
                    'Image channel, grayscale should be 1 and RGB should be 3')
FLAG.DEFINE_integer('start_epoch', 0, 'Number of epochs the trainer will run.')
FLAG.DEFINE_integer('epoch', 100, 'Number of epochs the trainer will run.')
FLAG.DEFINE_integer('batch_size', 64, 'Batch size.')
FLAG.DEFINE_integer('ckpt', 1, 'Save checkpoint every ? epochs.')
FLAG.DEFINE_integer('sample', 1, 'Get sample every ? epochs.')
FLAG.DEFINE_integer('summary', 100, 'Get summary every ? steps.')
FLAG.DEFINE_integer('gene_iter', 5,
                    'Train generator how many times every batch.')
FLAG.DEFINE_integer('disc_iter', 1,
                    'Train discriminator how many times every batch.')
FLAG.DEFINE_integer('gpu', 0, 'GPU No.')

TRAIN = False
VALIDATE = False
TEST = False
if FLAGS.mode == 'train':
  TRAIN = True
elif FLAGS.mode == 'validate':
  VALIDATE = True
elif FLAGS.mode == 'test':
  TEST = True
else:
  print('mode not recognized, process terminated.')
  exit()

BATCH_SIZE = FLAGS.batch_size
IMAGE_SIZE = FLAGS.image_size
IMAGE_CHANNEL = FLAGS.image_channel
HIDDEN_SIZE = int(IMAGE_SIZE / 2)

if IMAGE_SIZE == 128:
  from model_inpaint128 import generator
  from model_inpaint128 import discriminator
else:
  print('image_size not supported, process terminated.')
  exit()

DATANAME = FLAGS.dataname + str(IMAGE_SIZE)
DATA_PATH = os.path.join('data', DATANAME + '_' + FLAGS.mode + '.bin')
CHECKPOINT_DIR = os.path.join('checkpoints', DATANAME, 'gpu' + str(FLAGS.gpu))
LOG_DIR = os.path.join('logs', DATANAME, 'gpu' + str(FLAGS.gpu))
SAMPLE_DIR = os.path.join(
    'samples', DATANAME, 'gpu' + str(FLAGS.gpu), FLAGS.mode)

BETA1 = 0.5
BETA2 = 0.9
LAMB_REC = FLAGS.lamb_rec
LAMB_ADV = FLAGS.lamb_adv
LAMB_TV = FLAGS.lamb_tv
WEIGHT_DECAY_RATE = 0.00001


def run_model(sess):
  masked_image_holder = tf.placeholder(tf.float32, [
      BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name='masked_image')
  hidden_image_holder = tf.placeholder(tf.float32, [
      BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL], name='hidden_image')

  fake_image = generator(masked_image_holder, BATCH_SIZE,
                         IMAGE_CHANNEL, is_train=TRAIN, no_reuse=True)
  adv_real_score = discriminator(
      hidden_image_holder, BATCH_SIZE, is_train=TRAIN)
  adv_fake_score = discriminator(
      fake_image, BATCH_SIZE, reuse=True, is_train=TRAIN)
  adv_all_score = tf.concat([adv_real_score, adv_fake_score], axis=0)

  labels_disc = tf.concat(
      [tf.ones([BATCH_SIZE]), tf.zeros([BATCH_SIZE])], axis=0)
  labels_gene = tf.ones([BATCH_SIZE])
  correct = tf.equal(labels_disc, tf.round(tf.nn.sigmoid(adv_all_score)))
  disc_acc = tf.reduce_mean(tf.cast(correct, tf.float32))

  sampler = generator(masked_image_holder, BATCH_SIZE,
                      IMAGE_CHANNEL, is_train=False)

  if FLAGS.use_l1:
    rec_loss = tf.reduce_mean(
        tf.abs(tf.subtract(hidden_image_holder, fake_image)))
  else:
    rec_loss = tf.reduce_mean(
        tf.squared_difference(hidden_image_holder, fake_image))

  adv_disc_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_disc, logits=adv_all_score))
  adv_gene_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_gene, logits=adv_fake_score))

  tv_loss = tf.reduce_mean(tf.image.total_variation(fake_image))

  gene_loss_ori = LAMB_ADV * adv_gene_loss + \
      LAMB_REC * rec_loss + LAMB_TV * tv_loss
  disc_loss_ori = LAMB_ADV * adv_disc_loss

  all_vars = tf.trainable_variables()
  gene_vars = [var for var in all_vars if 'generator' in var.name]
  disc_vars = [var for var in all_vars if 'discriminator' in var.name]
  gene_weights = [var for var in gene_vars if 'weights' in var.name]
  disc_weights = [var for var in disc_vars if 'weights' in var.name]
  gene_loss = gene_loss_ori + WEIGHT_DECAY_RATE * \
      tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in gene_weights]))
  disc_loss = disc_loss_ori + WEIGHT_DECAY_RATE * \
      tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in disc_weights]))

  if TRAIN:
    if FLAGS.weight_clip:
      gene_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, BETA1)
      gene_vars_grads = gene_optimizer.compute_gradients(gene_loss, gene_vars)
      gene_vars_grads = [gv if gv[0] is None else [
          tf.clip_by_value(gv[0], -10., 10.), gv[1]] for gv in gene_vars_grads]
      gene_train_op = gene_optimizer.apply_gradients(gene_vars_grads)

      disc_optimizer = tf.train.AdamOptimizer(
          FLAGS.learning_rate / 10.0, BETA1)
      disc_vars_grads = disc_optimizer.compute_gradients(disc_loss, disc_vars)
      disc_vars_grads = [gv if gv[0] is None else [
          tf.clip_by_value(gv[0], -10., 10.), gv[1]] for gv in disc_vars_grads]
      disc_train_op = disc_optimizer.apply_gradients(disc_vars_grads)
    else:
      gene_train_op = tf.train.AdamOptimizer(
          FLAGS.learning_rate, BETA1, BETA2).minimize(gene_loss, var_list=gene_vars)
      disc_train_op = tf.train.AdamOptimizer(
          FLAGS.learning_rate, BETA1, BETA2).minimize(disc_loss, var_list=disc_vars)

    tf.summary.scalar('disc_acc', disc_acc)
    tf.summary.scalar('rec_loss', rec_loss)
    tf.summary.scalar('adv_gene_loss', gene_loss_ori)
    tf.summary.scalar('gene_loss', adv_gene_loss)
    tf.summary.scalar('disc_loss', adv_disc_loss)
    tf.summary.histogram('gene_score', adv_fake_score)
    tf.summary.histogram('disc_score', adv_all_score)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

  tf.global_variables_initializer().run()

  counter = 1
  saver = tf.train.Saver(max_to_keep=1)
  could_load, checkpoint_counter = load(sess, saver, CHECKPOINT_DIR)
  if could_load:
    counter = checkpoint_counter
    print(' [*] Load SUCCESS')
  else:
    print(' [!] Load FAILED...')
    if not TRAIN:
      exit()

  sess.graph.finalize()

  if TRAIN:
    save_configs()
    for epoch in range(FLAGS.start_epoch, FLAGS.epoch):
      index = 0
      losses = np.zeros(4)
      loss_file = open(os.path.join(LOG_DIR, 'loss_log.txt'), 'a+')
      file_object = open(DATA_PATH, 'rb')
      print('Current Epoch is: ' + str(epoch))
      for image_batch in read_by_batch(
              file_object, BATCH_SIZE, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]):
        if image_batch.shape[0] != BATCH_SIZE:
          break
        image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5
        if FLAGS.block == 'center':
          masked_image_batch, hidden_image_batch, masks_idx = preprocess_image(
              image_batch, BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL, False)
        else:
          masked_image_batch, hidden_image_batch, masks_idx = preprocess_image(
              image_batch, BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL)

        if epoch % FLAGS.sample == 0 and index == 0:
          samples, rec_loss_value, adv_gene_loss_value, adv_disc_loss_value = sess.run(
              [sampler, rec_loss, adv_gene_loss, adv_disc_loss],
              feed_dict={
                  masked_image_holder: masked_image_batch,
                  hidden_image_holder: hidden_image_batch
              })
          inpaint_image = np.copy(masked_image_batch)
          for idx in range(FLAGS.batch_size):
            idx_start1 = int(masks_idx[idx, 0])
            idx_end1 = int(masks_idx[idx, 0] + (HIDDEN_SIZE))
            idx_start2 = int(masks_idx[idx, 1])
            idx_end2 = int(masks_idx[idx, 1] + (HIDDEN_SIZE))
            inpaint_image[idx, idx_start1: idx_end1,
                          idx_start2: idx_end2, :] = samples[idx, :, :, :]

          save_images(image_batch, epoch, index, SAMPLE_DIR)
          save_images(inpaint_image, epoch, index + 1, SAMPLE_DIR)
          save_images(masked_image_batch, epoch, index + 2, SAMPLE_DIR)

          psnr, ssim = compute_psnr_ssim(hidden_image_batch, samples)
          # psnr, ssim = compute_psnr_ssim(image_batch, inpaint_image)
          print('[Getting Sample...] rec_loss:%.8f, gene_loss: %.8f, disc_loss: %.8f, \
psnr: %.8f, ssim: %.8f' % (rec_loss_value, adv_gene_loss_value,
                           adv_disc_loss_value, psnr, ssim))

        if counter % FLAGS.summary == 0:
          summary = sess.run(
              merged,
              feed_dict={
                  masked_image_holder: masked_image_batch,
                  hidden_image_holder: hidden_image_batch
              })
          writer.add_summary(summary, counter)

        if epoch % FLAGS.ckpt == 0 and index == 0:
          thread.start_new_thread(
              save, (sess, saver, CHECKPOINT_DIR, counter,))

        for _ in xrange(FLAGS.disc_iter):
          _ = sess.run(
              disc_train_op,
              feed_dict={
                  masked_image_holder: masked_image_batch,
                  hidden_image_holder: hidden_image_batch
              })

        for _ in xrange(FLAGS.gene_iter):
          _, rec_loss_value, adv_gene_loss_value, adv_disc_loss_value, disc_acc_value = sess.run(
              [gene_train_op, rec_loss, adv_gene_loss, adv_disc_loss, disc_acc],
              feed_dict={
                  masked_image_holder: masked_image_batch,
                  hidden_image_holder: hidden_image_batch
              })

        print('Epoch:%4d Batch:%4d, rec_loss:%2.8f, gene_loss: %2.8f, \
disc_loss: %2.8f, disc_acc: %2.8f' % (epoch, index, rec_loss_value, adv_gene_loss_value,
                                      adv_disc_loss_value, disc_acc_value))
        index += 1
        counter += 1
        losses[0] += rec_loss_value
        losses[1] += adv_gene_loss_value
        losses[2] += adv_disc_loss_value
        losses[3] += disc_acc_value

      losses /= index
      loss_file.write(
          'Epoch:%4d, rec_loss:%2.8f, gene_loss: %2.8f, disc_loss: %2.8f, disc_acc: %2.8f\n'
          % (epoch, losses[0], losses[1], losses[2], losses[3]))
  else:  # VALIDATE or TEST
    index = 0
    values = np.zeros(6)
    file_object = open(DATA_PATH, 'rb')
    for image_batch in read_by_batch(
            file_object, BATCH_SIZE, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]):
      if image_batch.shape[0] != BATCH_SIZE:
        break
      image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5
      if FLAGS.block == 'center':
        masked_image_batch, hidden_image_batch, masks_idx = preprocess_image(
            image_batch, BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL, False)
      else:
        masked_image_batch, hidden_image_batch, masks_idx = preprocess_image(
            image_batch, BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL)

      samples, rec_loss_value, adv_gene_loss_value, adv_disc_loss_value, disc_acc_value = sess.run(
          [sampler, rec_loss, adv_gene_loss, adv_disc_loss, disc_acc],
          feed_dict={
              masked_image_holder: masked_image_batch,
              hidden_image_holder: hidden_image_batch
          })
      inpaint_image = np.copy(masked_image_batch)
      for idx in range(FLAGS.batch_size):
        idx_start1 = int(masks_idx[idx, 0])
        idx_end1 = int(masks_idx[idx, 0] + HIDDEN_SIZE)
        idx_start2 = int(masks_idx[idx, 1])
        idx_end2 = int(masks_idx[idx, 1] + HIDDEN_SIZE)
        inpaint_image[idx, idx_start1: idx_end1,
                      idx_start2: idx_end2, :] = samples[idx, :, :, :]
      save_images(image_batch, index, 0, SAMPLE_DIR)
      save_images(inpaint_image, index, 1, SAMPLE_DIR)
      save_images(masked_image_batch, index, 2, SAMPLE_DIR)

      psnr, ssim = compute_psnr_ssim(hidden_image_batch, samples)
      # psnr, ssim = compute_psnr_ssim(image_batch, inpaint_image)
      print('[Getting Sample...] rec_loss:%2.8f, gene_loss: %2.8f, disc_loss: %2.8f, \
disc_acc: %2.8f, psnr: %2.8f, ssim: %2.8f' % (rec_loss_value, adv_gene_loss_value,
                                              adv_disc_loss_value, disc_acc_value, psnr, ssim))
      index += 1
      values[0] += rec_loss_value
      values[1] += adv_gene_loss_value
      values[2] += adv_disc_loss_value
      values[3] += disc_acc_value
      values[4] += psnr
      values[5] += ssim
    values /= index
    print('Mean rec_loss:%2.8f, gene_loss: %2.8f, disc_loss: %2.8f, disc_acc: %2.8f, \
psnr: %2.8f, ssim: %2.8f' % (values[0], values[1], values[2], values[3], values[4], values[5]))


def save_configs():
  config_file = open(os.path.join(CHECKPOINT_DIR, 'configs.txt'), 'a+')
  config_file.write('\
dataname:%s\n\
block:%s\n\
learning_rate:%f\n\
lamb_rec:%f\n\
lamb_adv:%f\n\
lamb_tv:%f\n\
use_l1:%s\n\
weight_clip:%s\n\
image_size:%d\n\
image_channel:%d\n\
start_epoch: %d\n\
epoch: %d\n\
batch_size: %d\n\
gene_iter: %d\n\
disc_iter: %d\n\n' % (FLAGS.dataname, FLAGS.block, FLAGS.learning_rate, FLAGS.lamb_rec,
                      FLAGS.lamb_adv, FLAGS.lamb_tv, str(FLAGS.use_l1),
                      str(FLAGS.weight_clip), FLAGS.image_size, FLAGS.image_channel,
                      FLAGS.start_epoch, FLAGS.epoch, FLAGS.batch_size, FLAGS.gene_iter,
                      FLAGS.disc_iter))


def print_args():
  print('dataname is:      ' + str(FLAGS.dataname))
  print('learning_rate is: ' + str(FLAGS.learning_rate))
  print('lamb_rec is:      ' + str(FLAGS.lamb_rec))
  print('lamb_adv is:      ' + str(FLAGS.lamb_adv))
  print('lamb_tv is:       ' + str(FLAGS.lamb_tv))
  print('start_epoch is:   ' + str(FLAGS.start_epoch))
  print('epoch is:         ' + str(FLAGS.epoch))
  print('image_size is:    ' + str(FLAGS.image_size))
  print('image_channel is: ' + str(FLAGS.image_channel))
  print('use_l1 is:        ' + str(FLAGS.use_l1))
  print('weight_clip is:   ' + str(FLAGS.weight_clip))
  print('mode is:          ' + str(FLAGS.mode))
  print('batch_size is:    ' + str(FLAGS.batch_size))
  print('ckpt is:          ' + str(FLAGS.ckpt))
  print('sample is:        ' + str(FLAGS.sample))
  print('summary is:       ' + str(FLAGS.summary))
  print('gene_iter is:     ' + str(FLAGS.gene_iter))
  print('disc_iter is:     ' + str(FLAGS.disc_iter))
  print('gpu is:           ' + str(FLAGS.gpu))
  print('')


def main(_):

  print_args()

  if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)

  run_config = tf.ConfigProto(allow_soft_placement=True)
  run_config.gpu_options.allow_growth = True
  with tf.device('/gpu:' + str(FLAGS.gpu)):
    with tf.Session(config=run_config) as sess:
      run_model(sess)


if __name__ == '__main__':
  tf.app.run()
