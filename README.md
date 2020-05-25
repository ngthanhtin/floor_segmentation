# floor-detection
This work based on MaskRCNN : https://github.com/matterport/Mask_RCNN. </br>

# Requirements
Tensorflow 1.14.</br>
Keras 2.0.8.</br>
Python 3.x.</br>

# Labeling
After using this [label tool](https://github.com/wkentaro/labelme), it will generate json files containing labels and vertexes inside your image folders. </br>
To train our model, first you need to convert json files to csv. Run `json2csv.py` to convert these to csv. To do this, you have to specify `root path` where stores your json files, name of the `exported csv` file and `column names` of this csv file.

# Usage
Firstly, you have to run `python setup.py install`. After that, you will train your model by the following instructions. </br>

# Training
Run `train.py` to train the model. </br>
These lines are to read data from exported csv files, you should change csv file path </br>
```
#Read csv files (read data)
segment_df_1 = pd.read_csv("abc_2.csv")
image_df_1 = segment_df_1.groupby('imagePath')['Pixels', 'Category'].agg(lambda x: list(x))
...
```
The model will load a pre-trained model called `mask_rcnn_coco.h5` to train on a new dataset. You can download this pre-trained model from this [link](https://github.com/matterport/Mask_RCNN/releases). </br>
The model will be trained by K-Fold method with different learning rates. </br>
You should define some configs in `config.py`. </br>

# Inference
Run `floor_segmentation_wrapper.py`. </br>