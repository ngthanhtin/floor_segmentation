import cv2
import random
import numpy as np
import pandas as pd
import ast
import os
import gc
import sys
import json
import glob
import matplotlib.pyplot as plt

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold

from config import *
from dataset import FloorDataset

floor_config = FloorConfig()
floor_config.display()

#Read csv files
segment_df_1 = pd.read_csv("abc_2.csv")
image_df_1 = segment_df_1.groupby('imagePath')['Pixels', 'Category'].agg(lambda x: list(x))

segment_df_2 = pd.read_csv("abc_3.csv")
image_df_2 = segment_df_2.groupby('imagePath')['Pixels', 'Category'].agg(lambda x: list(x))

segment_df_3 = pd.read_csv("abc_4.csv")
image_df_3 = segment_df_3.groupby('imagePath')['Pixels', 'Category'].agg(lambda x: list(x))

image_df = pd.concat([image_df_1, image_df_2, image_df_3])

dataset = FloorDataset()
dataset.add_data(df=image_df)
dataset.prepare()


#SHOW SOME IMAGE

for i in range(3):
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    black_white_img = visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)
    # cv2.imwrite('./black_white_img_4/color_img_{}.jpg'.format(i), black_white_img)
    # black_white_img = cv2.imread('./black_white_img_4/color_img_{}.jpg'.format(i), 0)
    # cv2.imwrite('./black_white_img_4/color_img_{}.jpg'.format(i), image)
    # cv2.imwrite('./black_white_img_4/img_{}.jpg'.format(i), ~black_white_img)
    
####-----------__TRAINING----------------------########
FOLD = 0
N_FOLDS = 5

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df) # ideally, this should be multilabel stratification

def get_fold():    
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]
        
train_df, valid_df = get_fold()

train_dataset = FloorDataset()
train_dataset.add_data(train_df)
train_dataset.prepare()

valid_dataset = FloorDataset()
valid_dataset.add_data(valid_df)
valid_dataset.prepare()


# train_segments = np.concatenate(train_df['ClassId'].values).astype(int)
print("Total train images: ", len(train_df))
# print("Total train segments: ", len(train_segments))

#train
LR = 1e-4
# EPOCHS = [2, 6, 8]
EPOCHS = [1,1,1]

import warnings 
warnings.filterwarnings("ignore")

model = modellib.MaskRCNN(mode='training', config=floor_config, model_dir=SAVE_MODEL_DIR)

model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

#augmentation
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  # only horizontal flip here
    iaa.GammaContrast((0.5, 1.5))
])

model.train(train_dataset, valid_dataset,
            learning_rate=LR*2, # train heads with higher lr to speedup learning
            epochs=EPOCHS[0],
            layers='heads',
            augmentation=None)

history = model.keras_model.history.history

model.train(train_dataset, valid_dataset,
            learning_rate=LR,
            epochs=EPOCHS[1],
            layers='all',
            augmentation=augmentation)

best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch - 1])

glob_list = glob.glob(f'./mask_rcnn_floor_{best_epoch:04d}.h5')
model_path = glob_list[0] if glob_list else ''


model.train(train_dataset, valid_dataset,
            learning_rate=LR/5,
            epochs=EPOCHS[2],
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

epochs = range(EPOCHS[-1])

print('End')
"""
plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()
plt.show()
"""

best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])