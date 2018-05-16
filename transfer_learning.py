#!/usr/bin/python

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

with tf.Session() as sess:
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

with open('codes', 'w') as f:
    codes.tofile(f)

import csv
with open('label', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)


from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)


from sklearn.model_selection import StratifiedShuffleSplit

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

train_idx, val_idx = next(ss.split(codes, labels))

half_val_len = int(len(val_idx)/2)
val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

train_x, train_y = codes[train_idx], labels_vecs[train_idx]
val_x, val_y = codes[val_idx], labels_vecs[val_idx]
test_x, test_y = codes[test_idx], labels_vecs[test_idx]

print "train: ", train_x.shape, train_y.shape
print "validation: ", val_x.shape, val_y.shape

inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])
fc = tf.contrib.layers.fully_connected(input_, 256)

logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer().minimize(cost)
predicted = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_batches(x, y, n_batches=10):
    batch_size = len(x)//n_batches
    
    for ii in range(0, n_batches*batch_size, batch_size):
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size] 
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y

epochs = 20
iteration = 0
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for x, y in get_batches(train_x, train_y):
            print x.shape,y.shape
            feed = {inputs_:x, labels_:y}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            print "epoch: ", epochs, "iteration:", iteration, "loss:", loss
            iteration += 1

            if iteration % 5 == 0:
                feed = {inputs_: val_x, labels_: val_y}
                val_acc = sess.run(accuracy, feed_dict=feed)
                print "validation:", val_acc


    saver.save(sess, "flowers.ckpt")

