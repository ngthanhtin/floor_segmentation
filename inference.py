import time
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from config import *
from utils import *

floor_config = tf.ConfigProto()
floor_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
floor_config.gpu_options.per_process_gpu_memory_fraction=0.2
floor_config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=floor_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def resize_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img

class FloorSegmentationWrapper:
    def __init__(self, seg_model_path = "./floor_segmentation.h5"):
        self.seg_model_path = seg_model_path
        self.person_detector = Person_Wrapper()

        self.inference_config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference', 
                          config=self.inference_config,
                          model_dir=self.seg_model_path)

        assert self.seg_model_path != '', "Provide path to trained weights"
        print("Loading weights from ", self.seg_model_path)
        self.model.load_weights(self.seg_model_path, by_name=True)

    def refine_masks(self, masks, rois):
        areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
        mask_index = np.argsort(areas)
        union_mask = np.zeros(masks.shape[:-1], dtype=bool)
        for m in mask_index:
            masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
            union_mask = np.logical_or(masks[:, :, m], union_mask)
        for m in range(masks.shape[-1]):
            mask_pos = np.where(masks[:, :, m]==True)
            if np.any(mask_pos):
                y1, x1 = np.min(mask_pos, axis=1)
                y2, x2 = np.max(mask_pos, axis=1)
                rois[m, :] = [y1, x1, y2, x2]
        return masks, rois
    
    def run_image(self, img):
        """
        Input: img was loaded.
        Output:masked image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = self.model.detect([resize_image(img)])        
        
        r = result[0]
        
        if r['masks'].size > 0:
            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(r['masks'].shape[-1]):
                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            y_scale = img.shape[0]/IMAGE_SIZE
            x_scale = img.shape[1]/IMAGE_SIZE
            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
            
            masks, rois = self.refine_masks(masks, rois)
        else:
            masks, rois = r['masks'], r['rois']

        t1 = time.time()
        masked_image = visualize.display_instances(img, rois, masks, r['class_ids'], ['bg']+label_names, r['scores'],title='segmentation', figsize=(12, 12))
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        print("Time visualize: ", time.time() - t1)
        return masked_image
    
    def run_video(self, video_path="./floor.mp4"):
        cap = cv2.VideoCapture(video_path)

        #write video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('./floor_segmentation.avi', cv2.VideoWriter_fourcc(* 'MJPG'), 10.0, (frame_width, frame_height))
        
        
        while (cap.isOpened()):
            ret, img = cap.read()
            masked_img = self.run_image(img)
            
            # write the img
            # out_2.write(masked_img)

            img = cv2.resize(img, (640, 480))            
            cv2.imshow("masked", masked_img)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    

if __name__ == "__main__":
    wrapper = FloorSegmentationWrapper()
    wrapper.run_video()