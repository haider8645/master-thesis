# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

# display plots in this notebook
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
<<<<<<< HEAD
caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
=======
caffe_root = '/home/haider/caffe/' # The caffe_root is changed to reflect the actual folder in the server.
>>>>>>> d7a01af7c50de886a0c011c0264d69eb6fdd7403
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe
#matplotlib inline
# set display defaults
# these are for the matplotlib figure's.
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
caffe.set_device(0)
caffe.set_mode_gpu()
# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
<<<<<<< HEAD
model_def = '/home/lod/master-thesis/examples/master-thesis/train-mnistCAE12sym0302.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/snapshots_iter_20000.caffemodel'
=======
model_def = '/home/haider/caffe/examples/master-thesis/train-mnistCAE12sym0302.prototxt'
model_weights = '/home/haider/caffe/examples/master-thesis/snapshots_iter_19000.caffemodel'
>>>>>>> d7a01af7c50de886a0c011c0264d69eb6fdd7403

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#bs = len(fnames)
#print bs
#img = cv2.imread('mnist_five.png', 0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img_blobinp = img[np.newaxis, np.newaxis, :, :]
#net.blobs['data'].reshape(*img_blobinp.shape)
#net.blobs['data'].data[...] = img_blobinp

<<<<<<< HEAD
dirname = '/home/lod/master-thesis/output_cae02'
=======
dirname = '/home/haider/caffe/output_cae02'
>>>>>>> d7a01af7c50de886a0c011c0264d69eb6fdd7403
#os.mkdir(dirname)
for j in range(15):
#os.path.join(dirname, face_file_name)
    net.forward()
    for i in range(1):
        cv2.imwrite(os.path.join(dirname,'image_image_' + str(j) + '.jpg'), 255*net.blobs['data'].data[0,i])

    for i in range(1):
        cv2.imwrite(os.path.join(dirname,'output_image_' + str(j) + '.jpg'), 255*net.blobs['deconv1neur'].data[0,i])



<<<<<<< HEAD
=======

>>>>>>> d7a01af7c50de886a0c011c0264d69eb6fdd7403
