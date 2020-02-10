import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
cust_dir    = 'data_snail-grain_crp'
sets=[\
    ('model_data/'+cust_dir, 'data_train'), \
    ('model_data/'+cust_dir,'data_val')\
    ]

classes = ["snail"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(p1, image_id, list_file):
    in_file     = open('model_data/{}/{}/{}.xml'.format(cust_dir, p1, image_id))
    out_file    = open('model_data/{}/{}_labels/{}.txt'.format(cust_dir, p1, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for p1, p2 in sets:
    imgs = list(filter(lambda x: '.jpg' in x, os.listdir(os.path.join(p1,p2)))) #Generate list of images
    image_ids = [img.split('.')[0] for img in imgs] # Strip off extension
    # image_ids = open('{}/{}.txt'.format(p1, p2)).read().strip().split()
    list_file = open('%s/%s.txt'%(p1, p2), 'w')
    for image_id in image_ids:
        list_file.write('{}/{}/{}/{}.jpg'.format(wd, p1, p2, image_id))
        convert_annotation(p2, image_id, list_file)
        # list_file.write('\n')
    list_file.close()

