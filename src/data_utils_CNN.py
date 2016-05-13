import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread

def load_food_image_batch(filename, num):
  """ load single batch of food images and labels """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    url_parts = datadict['Image URL'].split("/")
    img_fn    = url_parts[-1]
    with open(img_fn):
        X = f.read()
    Y = datadict['coarse_labels']
    X = X.reshape(num, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_scraped_food_images(ROOT):
  """ load all food images """
  Xtr, Ytr = load_food_image_batch(os.path.join(ROOT, 'train'),50000)
  Xte, Yte = load_food_image_batch(os.path.join(ROOT, 'test'),10000)
  return Xtr, Ytr, Xte, Yte
