# This script is to transform keras-yolo3 annotations into darknet annotations
# Main difference is: order of box arguments and relative coordinates
# keras-yolo3: 	<pixels_x> <pixels_y> <pixels_w> <pixels_h> <cls_id>
# darknet: 		<cls_id> <percnt_x> <percnt_y> <percnt_w> <percnt_h>
import sys
import argparse
from PIL import Image
import cv2
import os

def relative_boxes(fpath,boxes):
	im = Image.open(fpath)
	W = im.width
	H = im.height
	bxs = []
	for box in boxes:
		box = [int(item) for item in box.split(',')]
		x_min, y_min, x_max, y_max, obj = box
		w = x_max - x_min
		h = y_max - y_min
		# All object annotations in darknet are relative to the image WxH (0-100%)
		c_x = ( x_min + w/2 ) / W 										# centre-x is x_min + w/2
		c_y = ( y_min + h/2 ) / H 										# centre-y is y_min + h/2
		w = ( w ) / W
		h = ( h ) / H
		bx = [obj, c_x, c_y, w, h]
		bxs.append([str(item) if isinstance(item,int) else '{:.12f}'.format(item) for item in bx])
	return bxs
def main(args):
	# ds_name = input('dataset name: \n>')
	# ds_name = 'bedstraw_land'
	ds_name = args.ds_name
	dss_path = args.dss_path
	# calc_boxes = True if input('calc boxes? (Y/n)\n>').lower() == 'y' else False
	# draw_boxes = True if input('draw boxes? (Y/n)\n>').lower() == 'y' else False
	# calc_boxes, draw_boxes = False, True
	datasets = ['train', 'val']

	# Iterate over datasets requested
	for ds in datasets:
		# dss_path = 'model_data'
		ds_path = '{}/{}'.format(dss_path,ds_name)
		lpath = '{}/data_{}.txt'.format(ds_path,ds)
		pth = os.path.join(os.getcwd(), lpath)
		with open(pth,'r') as f:
			lines = f.readlines()

		train_list = []

		for line in [l.strip('\n') for l in lines]:
			objs = []
			parts = line.split('.')
			boxes = parts[1].replace('\r\n','').split(' ')
			fmt = boxes[0]
			boxes = boxes[1:]
			pname_ = parts[0].split('/')
			ppath, name = [ '/'.join(pname_[:-1]), pname_[-1] ]
			fname = name + '.' + fmt
			path = parts[0]
			fpath = path+'.'+fmt
			train_list.append(fpath)
			rel_boxes = relative_boxes(fpath, boxes)
			if args.calc_boxes:
				print('processing '+ fpath)
				with open(path+'.txt','w+') as l:
					print( str( len(boxes) ) + ' objects found: ')
					for i, box in enumerate(rel_boxes):
						box_line = ' '.join( box )
						print('box {}: '.format(i+1) + box_line)
						l.write( box_line + '\n')
			if args.draw_boxes:
				im = Image.open(fpath)
				W = im.width
				H = im.height
				im.close()
				im = cv2.imread(fpath)
				boxes_path = ppath + '/boxes'
				os.mkdir(boxes_path) if not os.path.exists(boxes_path) else None
				box_img_path = boxes_path + '/' + fname
				print('drawing onto '+ box_img_path +':')
				for box in rel_boxes:
					obj, c_x, c_y, w, h = [ float(itm) for itm in box ]
					vtx1 = ( round( (c_x - w/2) * W ) , round( (c_y - h/2) * H )  )
					vtx2 = ( round( (c_x + w/2) * W ) , round( (c_y + h/2) * H )  )
					cv2.rectangle(im, vtx1, vtx2, (255, 0, 0), 1)
					print('\t Obj: '+str(obj)+' between vertices '+ str(vtx1)+' and '+str(vtx2))
				cv2.imwrite( box_img_path, im )

		if args.calc_boxes:
			with open('{}/{}_list.txt'.format(ds_path,ds),'w+') as t:
				for img in train_list:
					# t.write( str(img.replace( '{}/data_{}'.format(ds_name,ds), 'images' )) + '\n' )
					t.write( str(img) + '\n' )

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
	parser.add_argument(
		'-b',
		'--draw-boxes',
		dest='draw_boxes',
		default=True,
		help='Whether to create a folder with boxes on images',
	)
	parser.add_argument(
		'-c',
		'--calc-boxes',
		dest='calc_boxes',
		default=True,
		help='Calculate boxes',
	)
	args = parser.parse_args()
	main(args)