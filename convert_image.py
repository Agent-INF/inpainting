import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

DATASET_NAME = 'voc_128'
IMAGE_SIZE = 128
IMAGE_CHANNEL = 3


def display_bin():
  file_object = open('data/' + DATASET_NAME + '._train.bin', 'rb')
  #file_object = open('data/' + DATASET_NAME + '_test.bin', 'rb')
  images = np.fromfile(file_object, dtype=np.uint8)
  images = np.reshape(images, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))
  print images.shape
  plt.figure('image')
  print images[0].shape
  if IMAGE_CHANNEL == 1:
    plt.imshow(images[100, :, :, 0], cmap='gray')
  elif IMAGE_CHANNEL == 3:
    plt.imshow(images[10])
  else:
    print 'image channel not supported'
  plt.show()


def convert_to_bin():
  dataset_path = 'data/' + DATASET_NAME + '/'
  path_pattern = dataset_path + '*.jpg'
  images = np.array(io.ImageCollection(path_pattern))
  np.random.shuffle(images)
  print images.shape
  num = images.shape[0]
  train_num = num * 4 / 5
  image_train = images[:train_num]
  image_test = images[train_num:]
  image_train.tofile('data/' + DATASET_NAME + '_train.bin')
  image_test.tofile('data/' + DATASET_NAME + '_test.bin')


def main():
  convert_to_bin()
  display_bin()


if __name__ == '__main__':
  main()
