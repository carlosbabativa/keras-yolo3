# Quick Training Session (From LabelImg images)
Note: Dependencies:
    -Tensorflow (1.12 tested OK)
    -Keras
    -Pillow
    -matplotlib
and
    yolov3[-tiny].weights converted to .h5
    You can obtain them with:
    ``` python convert.py {chosen_arch.cfg} {chosen_arch_pretrained.weights} {chosen_arch.h5} ```
    Save under at model_data with the name "yolov3.h5" or "yolov3-tiny.h5"
lastly
    Your dataset (images with xml annotations files) 
    e.g. as formatted by [labelImg](https://github.com/tzutalin/labelImg)
1. Create dataset folder in model_data 
<!-- and split data 80-20 % for training and validation -->
    model_data
    |--{dataset_name}
    |   |--data_train
    |       |--{images and xml labels}
<!-- 
    |   |--data_val
    |       |--{images and labels} -->

2. Update your classes' list in model_data/{dataset_folder}/labels.txt
<!-- 3. Modify my_annotation_bs.py (dataset_name) and run it. This outputs train and val txts with required input in dataset folder
    e.g.:
    {image1_path} [{{rect},{classID}}] -->
3. Generate yolo-style annotations ``` python  my_annotation.py -d {dataset_name} ``` 
    saved to data_train|val.txt in dataset folder
4. Calculate Anchors with ``` python kmeans.py -d {dataset_name} [-n {# of clusters, 6 if tiny}]``` 
<!-- 5. Create a copy and rename accoringly: yolov3-tiny_{dataset_suffix}.cfg
    5.1 Copy the line of anchors at model_data/{dataset_name}/calculated_anchors.txt
    5.2 Paste anchors in this file (on each yolo layer - 2 in total for yolov3-tiny)
    5.3 Calculate num_anchors * (5 + num_classes) and change value of filter in [convolutional] layer before each [yolo] layer
    [More Details](https://github.com/AlexeyAB/darknet/issues/4511) -->
<!-- 6. Convert pretrained weights with your customised cfg by running ``` python convert.py {your.cfg} {chosen_arch_pretrained.weights} model_data/{dataset_name}/cfg_name.h5``` -->
<!-- 7. Create a copy of train(caterp).py and rename accordingly -->
5. 0 Modify the config.json file to suit your trials
5. 1 Run ``` python my_train_general.py -d bedstraw_land_copies -c {your_config_set}.json ```

# TODO
[ ] Streamline steps as a single input of dataset folder path
    [ ] include annotations and anchor generation (kmeans) in train_general

---------------------
## ORIGINAL REPO:
# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
