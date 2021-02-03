import os
import tensorflow as tf
from tensorflow import keras
import random
from keras.callbacks import TensorBoard
#from keras import backend as K
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
from PIL import Image
from cxn_model import *
 
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
 
TRAIN_PATH_R = '/content/drive/My Drive/Colab Notebooks/dataset/B4/train/'  #change path 
TRAIN_PATH_G = '/content/drive/My Drive/Colab Notebooks/dataset/B3/train/'
TRAIN_PATH_B = '/content/drive/My Drive/Colab Notebooks/dataset/B2/train/'
 
TEST_PATH_R = '/content/drive/My Drive/Colab Notebooks/dataset/B4/test/'
TEST_PATH_G = '/content/drive/My Drive/Colab Notebooks/dataset/B3/test/'
TEST_PATH_B = '/content/drive/My Drive/Colab Notebooks/dataset/B2/test/'
 
X_train = np.zeros((350, IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((350, IMG_WIDTH, IMG_WIDTH, 1), dtype=np.float32)
img = np.zeros((IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
 
tr=np.zeros(350)
te=np.zeros(10)
 
for i in range(350):
       tr[i]=i;
for i in range(10):
       te[i]=i;
 
 
for n, id_ in tqdm(enumerate(tr),total=350):
       red    = Image.open(TRAIN_PATH_R + str(int(id_)) + '.png').convert('L')
       green  = Image.open(TRAIN_PATH_G + str(int(id_)) + '.png').convert('L')
       blue   = Image.open(TRAIN_PATH_B + str(int(id_)) + '.png').convert('L')
       
       rgb = Image.merge("RGB",(red,green,blue))
       img_b = np.asarray(rgb) 
       
       #img_r = imread(TRAIN_PATH_R + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
       #img_g = imread(TRAIN_PATH_G + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
       #img_b = imread(TRAIN_PATH_B + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
 
       #img_r = resize(img_r, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #img_g = resize(img_g, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
 
       img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
       img_b=img_b/255.0
       #for i in range(256):
       #       img[i] = np.concatenate((img_r[i],img_g[i],img_b[i]), axis=1)
       
       X_train[n] = img_b
       
       #mask = Image.open('/content/drive/My Drive/Colab Notebooks/dataset/BQA/train/' + str(int(102+id_)) + '.png').convert('L')
       mask = imread('/content/drive/My Drive/Colab Notebooks/dataset/BQA/train/' + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
       mask1 = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #mask1 = np.asarray(mask)
       #mask1 = resize(mask1, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
 
       Y_train[n] =mask1/255.0
       for io in range(256):
         for jo in range(256):
           if (Y_train[n][io][jo]>0.3):
             Y_train[n][io][jo]=1
           else:
             Y_train[n][io][jo]=0
       
 
# for test images 
 
X_test = np.zeros((10, IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_test = np.zeros((10, IMG_WIDTH, IMG_WIDTH, 1), dtype=np.float32)
img = np.zeros((IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
sizes_test = []
 
 
for n, id_ in tqdm(enumerate(te),total=10):
       red    = Image.open(TEST_PATH_R + str(170+int(id_)) + '.png').convert('L')
       green  = Image.open(TEST_PATH_G + str(170+int(id_)) + '.png').convert('L')
       blue   = Image.open(TEST_PATH_B + str(170+int(id_)) + '.png').convert('L')
       
       rgb = Image.merge("RGB",(red,green,blue))
       img_b = np.asarray(rgb)  
 
       img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
       img_b=img_b/255.0
       #for i in range(256):
       #       img[i] = np.concatenate((img_r[i],img_g[i],img_b[i]), axis=1)
       
       X_test[n] = img_b
 
       mask = imread('/content/drive/My Drive/Colab Notebooks/dataset/BQA/test/' + str(int(170+id_)) + '.png')[:,:,:IMG_CHANNELS]
       mask1 = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #mask1 = np.asarray(mask)
       #mask1 = resize(mask1, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
 
       Y_test[n] =mask1/255.0
       for io in range(256):
         for jo in range(256):
           if (Y_test[n][io][jo]>0.3):
             Y_test[n][io][jo]=1
           else:
             Y_test[n][io][jo]=0


model = model_arch(input_rows=256, input_cols=256, num_of_channels=3, num_of_classes=1)
model.compile(optimizer = Adam(lr = 1e-4), loss = jacc_coef, metrics = [jacc_coef,'accuracy'])


#checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5' , verbose=1, save_best_only=True)
#callbacks = [tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'),tf.keras.callbacks.TensorBoard(log_dir="logs")]


results = model.fit(X_train, Y_train, validation_split=0.05, batch_size=2, epochs=70, verbose=1)   #, callbacks=[cp_callback])


preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.95):], verbose=1)
preds_test = model.predict(X_test, verbose=1)


preds_train_t = (preds_train > 0.5).astype(np.float32)
preds_val_t = (preds_val > 0.5).astype(np.float32)
preds_test_t = (preds_test > 0.5).astype(np.float32)

train_acc = model.evaluate(X_train, Y_train, verbose=1)
test_acc = model.evaluate(X_test, Y_test, verbose=1)

