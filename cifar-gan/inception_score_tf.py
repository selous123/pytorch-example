'''
From https://github.com/tsc2017/inception-score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py
Args:
    images: A numpy array with values ranging from -1 to 1 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary.
    splits: The number of splits of the images, default is 10.
Returns:
    mean and standard deviation of the inception across the splits.
'''

import tensorflow as tf
import os, sys
import functools
import numpy as np
import math
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
import cPickle
tfgan = tf.contrib.gan

#set tensorflow gpu source
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session=tf.InteractiveSession(config=config)

BATCH_SIZE=64

# Run images through Inception.
inception_images=tf.placeholder(tf.float32,[BATCH_SIZE,3,None,None])
def inception_logits(images=inception_images, num_splits=1):
    images=tf.transpose(images,[0,2,3,1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(
    images, num_or_size_splits=num_splits)
    logits = functional_ops.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='logits:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

logits=inception_logits()

def get_inception_probs(inps):
    preds = []
    n_batches = len(inps)//BATCH_SIZE
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        pred = logits.eval({inception_images:inp})[:,:1000]
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    preds=np.exp(preds)/np.sum(np.exp(preds),1,keepdims=True)
    return preds

def preds2score(preds,splits):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def inception_score(images, splits=10):
    assert(type(images) == np.ndarray)
    assert(len(images.shape)==4)
    assert(images.shape[1]==3)
    assert(np.max(images[0])<=1)
    assert(np.min(images[0])>=-1)

    start_time=time.time()
    preds=get_inception_probs(images)
    print ('Inception Score for %i samples in %i splits'% (preds.shape[0],splits))
    mean,std = preds2score(preds,splits)
    print 'Inception Score calculation time: %f s'%(time.time()-start_time)
    return mean,std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits (default).


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = cPickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

if __name__=='__main__':
    file_name = '/home/lrh/dataset/cifar-10/cifar-10-batches-py'
    xtr,ytr,xte,yte = load_CIFAR10(file_name)
    xtr = (xtr.transpose(0,3,1,2) - 128) / 128.0
    print inception_score(xtr)
