#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append("/home/morfast/flower/")

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

data_dir = 'flower_photos/'
contents = os.listdir(data_dir)
print contents
# name of flower types
classes = [each for each in contents if os.path.isdir(data_dir + each)]

print classes

batch_size = 10
codes_list = []
labels = []
batch = []
codes = None

with tf.Session() as sess, tf.device('/gpu:0'):
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224,224,3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)

    for flower_class in classes:
        print "starting %s images" % (flower_class)
        class_path = data_dir + flower_class
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(flower_class)

            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)

                feed_dict = {input_: images}
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))


                batch = []
        print "%d images processed" % (ii)


print len(codes), len(labels)

# save to files
import pickle
f = open("codes", "wb")
pickle.dump(codes, f)
f.close()

f = open("label", "wb")
pickle.dump(labels, f)
f.close()

