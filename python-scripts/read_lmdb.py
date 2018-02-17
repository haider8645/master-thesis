import lmdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import sys
import caffe

# Make sure that caffe is on the python path:

caffe_root = '/home/haider/caffe'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
db_path = '/home/haider/caffe/LMDB-datasets/train_lmdb/'
dirname = '/home/haider/caffe/LMDB-datasets/'

for i in range(1):
    x = '00007'
    lmdb_env = lmdb.open(db_path)  # equivalent to mdb_env_open()
    lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
    lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
#print lmdb_cursor.first()
#print lmdb_cursor.key()
#print lmdb_cursor.get('00001')
    lmdb_cursor.get(x) #  get the data associated with the 'key' 1, change the value to get other images
    value = lmdb_cursor.value()
    key = lmdb_cursor.key()
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    image = np.zeros((datum.channels, datum.height, datum.width))
    image = caffe.io.datum_to_array(datum)
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, (2, 1, 0)]
    image = image.astype(np.uint8)
    cv2.imwrite(os.path.join(dirname,'output_image_2' + str(i) + '.png'), image)
