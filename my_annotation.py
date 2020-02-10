import xml.etree.ElementTree as ET
from os import getcwd
import os


cust_dir    = 'data_snail-grain_crp'
train_dir   = 'model_data/{}/data_train'.format(cust_dir)
val_dir     = 'model_data/{}/data_val'.format(cust_dir)
if not os.path.exists(train_dir + '.txt'):
    open(train_dir+'.txt', 'w+').write('\n'.join(os.listdir(train_dir)))
    open( val_dir+ '.txt', 'w+').write('\n'.join(os.listdir( val_dir )))
# os.system('gen_img_lists.bat')
sets=[('model_data/'+cust_dir, 'data_train'), ('model_data/'+cust_dir,'data_val')]

classes = ["snail"]


def convert_annotation(p1, image_id, list_file):
    in_file = open('model_data/{}/{}/{}.xml'.format(cust_dir,p1, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for p1, p2 in sets:
    imgs = list(filter(lambda x: '.jpg' in x, os.listdir(os.path.join(p1,p2)))) #Generate list of images
    image_ids = [img.split('.')[0] for img in imgs] # Strip off extension
    # image_ids = open('{}/{}.txt'.format(p1, p2)).read().strip().split()
    list_file = open('%s/%s.txt'%(p1, p2), 'w')
    for image_id in image_ids:
        list_file.write('{}/{}/{}/{}.jpg'.format(wd, p1, p2, image_id))
        convert_annotation(p2, image_id, list_file)
        list_file.write('\n')
    list_file.close()

