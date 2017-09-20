# inpainting
Another Implementation of "Context Encoders: Feature Learning by Inpainting"

Use convert_image.py to convert all jpg images in a folder into one or more binary files, which will use for training or validating or testing.

Then run "python inpaint.py" and the training will started.

The program will save sample images every epoch in ./samples folder by default.

I run my program in Ubuntu 16.04, Tensorflow 1.2 and python 2.7.

PS: I've made lots of changes compare to jazzsaxmafia/Inpainting because this is part of my private project, but it will do the inpainting job. 
