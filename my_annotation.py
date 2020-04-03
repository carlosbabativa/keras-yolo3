import sys
import argparse
import xml.etree.ElementTree as ET
from os import getcwd
import os

def convert_annotation(p1, image_id, list_file, ds_name, classes):
    global datasets_path
    in_file = open('{}/{}/{}/{}.xml'.format(datasets_path,ds_name,p1, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        difficult_val = difficult.text if difficult is not None else 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult_val)==1 or difficult is not None:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def main(ds_name, dss_path):
    global datasets_path 
    datasets_path = dss_path if dss_path != 'model_data' else os.path.abspath(os.getcwd()) +'/' + dss_path
    list_args = datasets_path + ' ' + ds_name
    os.system('gen_img_lists.bat '+ list_args) if os.name == 'nt' else os.system('bash gen_img_lists.sh '+list_args + ' 2> /dev/null')
    ds_r_path = datasets_path+'/'+ds_name
    sets=[ (ds_r_path,'data_train'), (ds_r_path,'data_val') ]
    # for ds, subds in sets:
    #     listpath = ds + '/' + subds
    #     with open(listpath+'.txt','w+') as f:
    #         flist = os.listdir(listpath)
    #         imglist = list(filter(lambda f:'.jpg' in f or '.png' in f, flist))
    #         strimgls = '/n'.join(['/'.join((os.getcwd(),ds,img)) for img in imglist])
    #         f.write(strimgls)
    classes = [f.strip('\n') for f in open('{}/{}/labels.txt'.format(datasets_path,ds_name))]
    # wd = getcwd()

    for p1, p2 in sets:
        imgs = list(filter(lambda x: '.jpg' in x, os.listdir(os.path.join(p1,p2)))) #Generate list of images
        image_ids = [img.split('.')[0] for img in imgs] # Strip off extension
        # image_ids = open('{}/{}.txt'.format(p1, p2)).read().strip().split()
        list_file = open('%s/%s.txt'%(p1, p2), 'w')
        for image_id in image_ids:
            list_file.write('{}/{}/{}.jpg'.format(p1, p2, image_id))
            convert_annotation(p2, image_id, list_file, ds_name, classes)
            list_file.write('\n')
        list_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset',
        dest='ds_name',
        required=True,
        help='Name of dataset (folder in datastes folder \'data\')',
    )
    parser.add_argument(
        '-p',
        '--datasets-path',
        dest='dss_path',
        default='model_data',
        help='Name of dataset (folder in datastes folder \'data\')',
    )
    args = parser.parse_args()
    main(args.ds_name, args.dss_path)