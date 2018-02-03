# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

# display plots in this notebook
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
caffe_root = '/home/haider/caffe/' # The caffe_root is changed to reflect the actual folder in the server.
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
model_def = '/home/haider/caffe/examples/master-thesis/train-mnistCAE12sym0302.prototxt'
model_weights = '/home/haider/caffe/examples/master-thesis/snapshots_iter_20000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


print("Network layers:")
for name, layer in zip(net._layer_names, net.layers):
    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))
x_data_plot = []
y_data_plot = []
labels=[]

#fig = plt.figure()
#ax = fig.add_subplot(1,1,1,axisbg = "1.0")
for j in range(10):
    net.forward()
    X = net.blobs["ip2encode"].data[0]
#print("output")
   # if net.blobs["label"].data[0] == 1:
   #     print "ONE"
    T=net.blobs["label"].data[0]
#    labels.append(T[0])
#    print T[0]
    print T[0]
    if T[0] == 0:
        plt.scatter(X[0],X[1],s=50,c="black",alpha=0.8,label="zero")
    if T[0] == 1:
        plt.scatter(X[0],X[1],s=50,c="red",alpha=0.8,label="one")
    if T[0] == 2:
        plt.scatter(X[0],X[1],s=50,c="gold",alpha=0.8,label="two")
    if T[0] == 3:
        plt.scatter(X[0],X[1],s=50,c="blue",alpha=0.8,label="three")
    if T[0] == 4:
        plt.scatter(X[0],X[1],s=50,c="green",alpha=0.8,label="four")
    if T[0] == 5:
        plt.scatter(X[0],X[1],s=50,c="orange",alpha=0.8,label="five")
    if T[0] == 6:
        plt.scatter(X[0],X[1],s=50,c="magenta",alpha=0.8,label="six")
    if T[0] == 7:
        plt.scatter(X[0],X[1],s=50,c="pink",alpha=0.8,label="seven")
    if T[0] == 8:
        plt.scatter(X[0],X[1],s=50,c="brown",alpha=0.8,label="eight")
    if T[0] == 9:
        plt.scatter(X[0],X[1],s=50,c="yellow",alpha=0.8,label="nine")
    x_data_plot.append(X[0])
    y_data_plot.append(X[1])

labels = ("zero","one","two","three","four","five","six","seven","eight","nine")
colors = ("black","red","gold","blue","indigo","orange","darkblue","crimson","lightpink","olive")
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

#for data, color, group in zip(data, colors, labels):
#    x, y = data
#    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

#plt.title('Matplot scatter plot')
#plt.legend(loc=2)
#plt.show()


#plt.scatter(x_data_plot,y_data_plot,s=50,c=colors,alpha = 0.8)
#print X[0]
#print X[1]
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1,axisbg="1.0")
#for data_x,data_y,lab in zip(x_data_plot,y_data_plot,labels):
#ax.scatter(data_x,data_y,label=lab)
plt.show()

#plt.plot(x_data_plot,y_data_plot,'ro')
#plt.show()



#bs = len(fnames)
#print bs
#img = cv2.imread('mnist_five.png', 0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img_blobinp = img[np.newaxis, np.newaxis, :, :]
#net.blobs['data'].reshape(*img_blobinp.shape)
#net.blobs['data'].data[...] = img_blobinp
#dirname = '/home/lod/master-thesis/graphs/output_cae02'
#os.mkdir(dirname)
#for j in range(10):
#os.path.join(dirname, face_file_name)
#    net.forward()
#    for i in range(1):
#        cv2.imwrite(os.path.join(dirname,'input_image_' + str(j) + '.jpg'), 255*net.blobs['data'].data[0,i])

#    for i in range(1):
#        cv2.imwrite(os.path.join(dirname,'output_image_' + str(j) + '.jpg'), 255*net.blobs['deconv1neur'].data[0,i])
