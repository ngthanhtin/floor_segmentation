from mrcnn.config import Config

label_names = ['floor'] # label classes
NUM_CATS = len(label_names)
IMAGE_SIZE = 512 # size of image after resizing

COCO_WEIGHTS_PATH = './mask_rcnn_coco.h5'
SAVE_MODEL_DIR = './floor_mask_rcnn_model/'

# Config
class FloorConfig(Config):
    NAME = "floor"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200

class InferenceConfig(ClotheConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1