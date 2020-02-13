"""
Retrain the YOLO model for your own dataset.
"""
import sys
import json
import argparse

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main(ds_name, train_trial_name, train_cfg):
    global base_model
    stg_epochs, batches, base_model = fetch_config(train_cfg)
    
    annotation_path = 'model_data/{}/data_train.txt'.format(ds_name)
    log_dir = 'logs/{}/'.format(train_trial_name)
    classes_path = 'model_data/{}/labels.txt'.format(ds_name)
    anchors_path = 'model_data/{}/calculated_anchors.txt'.format(ds_name)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/{}.h5'.format(base_model))
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.25
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # hist = []
    has_stg3 = len(batches) > 2
    #STAGE 1 
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(
            optimizer=Adam(lr=1e-3), 
            loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred},
            metrics=['accuracy']
        )

        batch_size = batches[0]
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        # hist.append(
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=stg_epochs[0],
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        # )
        model.save_weights('model_data/{}_stage_1.h5'.format(train_trial_name))

    #STAGE 2
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}, metrics=['accuracy']) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = batches[1] # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=stg_epochs[1],
        initial_epoch=stg_epochs[0],
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        # )
        stg2_sfx = 'stage_2' if has_stg3 else 'final'
        model.save_weights('model_data/{}_{}.h5'.format(train_trial_name,stg2_sfx))

    # Further training if needed.
    # STAGE 3
    if has_stg3:
        batch_size = batches[2] # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        # hist.append(
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=stg_epochs[2],
        initial_epoch=stg_epochs[1],
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        # )
        model.save_weights('model_data/{}_final.h5'.format(train_trial_name))
    #Write results
    # with open('model_data/trials.txt','a+') as f:
    #     f.write('\nResults for trial: ' + train_trial_name + '\n')
        # f.write('')
        # f.write('Final loss: {} \t\t {} \t\t '+'{}'.format(hist[0].losses[-10:],hist[1].losses[-10:]))



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolov3-tiny.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes, based on {}.'.format(num_anchors, num_classes,weights_path))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def fetch_config(train_cfg):
    default = {
        'stg_epochs':[50,100],
        'batches'   :[12,32],
        'base'      :'yolov3-tiny'
    }
    for k in default.keys():
        if k not in train_cfg.keys():
            train_cfg[k] = default[k]
    new = {k:nv if v != nv else v for k,v,nv in zip(default.keys(),default.values(),train_cfg.values())}
    # new = {k:tc[k] if tc[k] != df[k] else df[k] for k in }
    return (new.values())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--dataset',
        dest='ds_name',
        required=True,
        help='Name of dataset (folder in datastes folder \'data\')',
    )
    parser.add_argument(
        '-c',
        '--configs',
        dest='cfgs',
        help=
        '''
        Training configuration for output models
        pass a json with the following format:
        {
            'train_cfg_1': {
                'stg_epochs':[50,100],
                'batches'   :[3,6],
                'base'      :'yolov3-tiny'
            },
            'train_cfg_2': {
                'stg_epochs':[50, 120, 150],
                'batches'   :[2, 3, 6],
                'base'      :'train_cfg_1'
            }
        }
        '''
    )

    args = parser.parse_args()
    with open(args.cfgs,'r') as j:
        configs = json.load(j)
    for train_trial, train_cfg in configs.items():
        ds_name = args.ds_name
        _main(ds_name, train_trial, train_cfg) 
