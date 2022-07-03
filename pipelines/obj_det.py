from cmath import inf
import os
import glob
import time
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
from keras.utils import np_utils
import torch
import datetime

from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')

from pipeline_input import *
from constants import *

class KITTI_lemenko_interp(pipeline_dataset_interpreter):

	def load_calibration(self, calib_file_name):
		calib = {}
		f = open(calib_file_name, "r")
		for line in f:
			key_val = line.split(":")
			if len(key_val)==2:
				key, value = key_val
				if key!='' and value!='':
					value = np.array(list(map(float, value.strip().split(" "))))
					calib[key] = value
			elif len(key_val)==1:
				pass # Last line, no issues
			else:
				raise Exception("Malformed Calib file: " + calib_file_name)
		return calib

	def load_labels(self, label_file_name):
		return pd.read_csv(label_file_name, sep=" ", 
                       names=['label', 'truncated', 'occluded', 'alpha', 
                              'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 
                              'bbox_ymax', 'dim_height', 'dim_width', 'dim_length', 
                              'loc_x', 'loc_y', 'loc_z', 'rotation_y'])
		

	def load(self) -> None:
		print("Loading KITTI_lemenko_interp from:", self.input_dir)
		data_object_calib = os.path.join(self.input_dir,"data_object_calib")
		data_object_image_2 = os.path.join(self.input_dir,"data_object_image_2")
		data_object_image_3 = os.path.join(self.input_dir,"data_object_image_3")
		data_object_label_2 = os.path.join(self.input_dir,"data_object_label_2")

		assert os.path.exists(data_object_calib), data_object_calib
		assert os.path.exists(data_object_image_2), data_object_image_2
		assert os.path.exists(data_object_image_3), data_object_image_3
		assert os.path.exists(data_object_label_2), data_object_label_2

		dataset = {}

		for mode in ('testing', 'training'):
			dataset[mode] = {
				'image_2': [], 'image_3':[], 'calib': [], 'label_2': []
			}
			image_2_files_list = sorted(glob.glob(os.path.join(data_object_image_2, mode, "image_2", "*.png")))
			files_list = list(map(lambda x: x.split("/")[-1].split(".png")[:-1][0], image_2_files_list))
			#print(files_list)
			print("KITTI_lemenko_interp: load", mode)
			for f in tqdm(files_list):
				calib_path = os.path.join(data_object_calib, mode, "calib", f+".txt")
				image_2_path = os.path.join(data_object_image_2, mode, "image_2", f+".png")
				image_3_path = os.path.join(data_object_image_3, mode, "image_3", f+".png")
				label_2_path = os.path.join(data_object_label_2, mode, "label_2", f+".txt")
					
				assert os.path.exists(calib_path), calib_path
				assert os.path.exists(image_2_path), image_2_path
				assert os.path.exists(image_3_path), image_3_path
				if mode=="training":
					assert os.path.exists(label_2_path), label_2_path
				
				dataset[mode]['calib'] += [self.load_calibration(calib_path)]
				dataset[mode]['image_2'] += [image_2_path]
				dataset[mode]['image_3'] += [image_3_path]
				if mode=='training':
					dataset[mode]['label_2'] += [self.load_labels(label_2_path)]
				else:
					dataset[mode]['label_2'] += [None]
			
			dataset[mode] = pd.DataFrame(dataset[mode])
		
		xtrain, xtest = self.generate_data(dataset['training'])
		self.dataset = {
			'train': {
				'x': xtrain["image"].unique(),
				'y': xtrain
			},
			'test': {
				'x': xtest["image"].unique(),
				'y': xtest
			}
		}


	def generate_data(self, dataset):
		information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'name':[] ,'label':[], 'image':[]}
		for index, row in dataset.iterrows():
			kitti_labels = row['label_2']
			for index2, row2 in kitti_labels.iterrows():
				label = "None"
				if row2['label'] == 'Pedestrian':
					label = "person"
				# elif row2['label'] == 'Car':
				# 	label = "car"
				else:
					label = "DontCare"

				#if label != "None":
				if True:
					information['xmin']+=[row2['bbox_xmin']]
					information['ymin']+=[row2['bbox_ymin']]
					information['xmax']+=[row2['bbox_xmax']]
					information['ymax']+=[row2['bbox_ymax']]

					#information['name']+=[row['image_2'].split("/")[-1]]
					information['name']+=[row['image_2']]
					#information['label']+=[label]
					information['label']+=[row2['label']]
					information['image']+=[row['image_2']]

		information = pd.DataFrame(information)
		#return information, information
		return train_test_split(information, test_size=0.1, random_state=1)


class obj_det_interp_1(pipeline_dataset_interpreter):
	def load(self) -> None:
		super().load()
		train_path=os.path.join(self.input_dir, 'Train/Train/JPEGImages')
		train_annot=os.path.join(self.input_dir, 'Train/Train/Annotations')
		test_path=os.path.join(self.input_dir, 'Test/Test/JPEGImages')
		test_annot=os.path.join(self.input_dir, 'Test/Test/Annotations')
		assert os.path.exists(train_path)
		assert os.path.exists(train_annot)
		assert os.path.exists(test_path)
		assert os.path.exists(test_annot)
		xtrain = self.generate_data(train_annot, train_path)
		xtest = self.generate_data(test_annot, test_path)
		self.dataset = {
			'train': {
				'x': xtrain["image"].unique(),
				'y': xtrain
			},
			'test': {
				'x': xtest["image"].unique(),
				'y': xtest
			}
		}
		
	def generate_data(self, Annotpath, Imagepath):
		information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[],'name':[] ,'label':[], 'image':[]}
		for file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
			dat=ET.parse(file)
			for element in dat.iter():    
				if 'object'==element.tag:
					for attribute in list(element):
						if 'name' in attribute.tag:
							name = attribute.text
							file_name = file.split('/')[-1][0:-4]
							information['label'] += [name]
							information['name'] +=[file_name]
							#information['name'] +=[file]
							information['image'] += [os.path.join(Imagepath, file_name + '.jpg')]
						if 'bndbox'==attribute.tag:
							for dim in list(attribute):
								if 'xmin'==dim.tag:
									xmin=int(round(float(dim.text)))
									information['xmin']+=[xmin]
								if 'ymin'==dim.tag:
									ymin=int(round(float(dim.text)))
									information['ymin']+=[ymin]
								if 'xmax'==dim.tag:
									xmax=int(round(float(dim.text)))
									information['xmax']+=[xmax]
								if 'ymax'==dim.tag:
									ymax=int(round(float(dim.text)))
									information['ymax']+=[ymax]
		return pd.DataFrame(information)

class obj_det_data_visualizer(pipeline_data_visualizer):

	def visualize(self, x, y, preds, save_dir) -> None:
		plot = True
		image_names_list = y["name"].unique()
		iou_list = []
		iou_thresh = 0.5
		yolo_metrics = {
			'tp':0, 	# iou>thresh
			'fp': 0, 	# 0<iou<thresh
			'fn':0		# iou==0	
		}
		print("obj_det_data_visualizer: visualize")
		for image_name in tqdm(image_names_list):
			iou_list = []
			labels = y[y["name"]==image_name]
			detections = preds[preds["name"]==image_name]
			for index1, lab in labels.iterrows():
				largest_iou = 0.0
				for index2, yolo_bb in detections.iterrows():
					iou = get_iou(lab, yolo_bb)
					if iou > largest_iou:
						largest_iou = iou
				if largest_iou==0:
					yolo_metrics['fn'] += 1
				else:
					if largest_iou>iou_thresh:
						yolo_metrics['tp'] += 1
					else:
						yolo_metrics['fp'] += 1
				iou_list.append(largest_iou)

			image_path = labels["image"].iloc[0]
			img = cv2.imread(image_path)
			for index1, lab in labels.iterrows():
				img = cv2.rectangle(img, (round(lab['xmin']), round(lab['ymin'])), (round(lab['xmax']), round(lab['ymax'])), (255,255,0),2)
			for index2, lab in detections.iterrows():
				img = cv2.rectangle(img, (round(lab['xmin']), round(lab['ymin'])), (round(lab['xmax']), round(lab['ymax'])), (0,255,0),2)
				# print(len(labels), len(detections))
				# print(labels)
				# print(detections)
			save_path = os.path.join(save_dir, str(datetime.datetime.now()).replace(" ", "_") + ".png")
			#cv2.imshow("img", img)
			#cv2.waitKey(0)
			#time.sleep(0.5)
			if min(iou_list)<0.5:
				cv2.imwrite(save_path, img)

class obj_det_evaluator:

	def evaluate(self, x, y, plot=False):
		preds = self.predict(x)
		image_names_list = y["name"].unique()
		iou_list = []
		iou_thresh = 0.5
		yolo_metrics = {
			'tp':0, 	# iou>thresh
			'fp': 0, 	# 0<iou<thresh
			'fn':0		# iou==0	
		}
		print("obj_det_evaluator")
		for image_name in tqdm(image_names_list):
			labels = y[y["name"]==image_name]
			detections = preds[preds["name"]==image_name]
			for index1, lab in labels.iterrows():
				largest_iou = 0.0
				for index2, yolo_bb in detections.iterrows():
					iou = get_iou(lab, yolo_bb)
					if iou > largest_iou:
						largest_iou = iou
				if largest_iou==0:
					yolo_metrics['fn'] += 1
				else:
					if largest_iou>iou_thresh:
						yolo_metrics['tp'] += 1
					else:
						yolo_metrics['fp'] += 1
				iou_list.append(largest_iou)
			if plot:
				image_path = labels["image"].iloc[0]
				img = cv2.imread(image_path)
				for index1, lab in labels.iterrows():
					img = cv2.rectangle(img, (round(lab['xmin']), round(lab['ymin'])), (round(lab['xmax']), round(lab['ymax'])), (255,0,0),2)
				for index2, lab in detections.iterrows():
					img = cv2.rectangle(img, (round(lab['xmin']), round(lab['ymin'])), (round(lab['xmax']), round(lab['ymax'])), (0,255,0),2)
				print(len(labels), len(detections))
				print(labels)
				print(detections)

		prec = np.float64(yolo_metrics['tp']) / float(yolo_metrics['tp'] + yolo_metrics['fp'])
		recall = np.float64(yolo_metrics['tp']) / float(yolo_metrics['tp'] + yolo_metrics['fn'])
		f1_score = np.float64(2*prec*recall)/(prec+recall)
		iou_avg = np.float64(sum(iou_list)) / len(iou_list)
		results = {
			'prec': prec,
			'recall': recall,
			'f1_score': f1_score,
			'iou_avg': iou_avg,
			'confusion': yolo_metrics
		}
		print(results)
		return results, preds


class obj_det_pipeline_model(obj_det_evaluator, pipeline_model):

	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
		
	def train(self, x, y) -> np.array:
		preds = self.predict(x)
		image_names_list = y["name"].unique()
		
		results = {
			'training_results': 0,
		}
		return results, preds

		
	def predict(self, x: dict) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		predict_results = {
			'xmin': [], 'ymin':[], 'xmax':[], 'ymax':[], 'confidence': [], 'name':[], 'image':[]
		}
		print("obj_det_pipeline_model: predict")
		for image_path in tqdm(x):
			img = cv2.imread(image_path)
			results = self.model(image_path)
			df = results.pandas().xyxyn[0]
			res = df[df["name"]=="person"]
			for index, yolo_bb in res.iterrows():
				file_name = image_path.split('/')[-1][0:-4]
				predict_results["xmin"] += [yolo_bb["xmin"]*img.shape[1]]
				predict_results["ymin"] += [yolo_bb["ymin"]*img.shape[0]]
				predict_results["xmax"] += [yolo_bb["xmax"]*img.shape[1]]
				predict_results["ymax"] += [yolo_bb["ymax"]*img.shape[0]]
				predict_results["confidence"] += [yolo_bb["confidence"]]
				predict_results["name"] += [file_name]
				predict_results["image"] += [image_path]
		predict_results = pd.DataFrame(predict_results)
		return predict_results


class obj_det_pipeline_model_yolov5n(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
		
class obj_det_pipeline_model_yolov5s(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

class obj_det_pipeline_model_yolov5m(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

class obj_det_pipeline_model_yolov5l(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

class obj_det_pipeline_model_yolov5x(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

class obj_det_pipeline_ensembler_1(obj_det_evaluator, pipeline_ensembler):

	def predict(self, x: dict) -> np.array:
		model_names = list(x.keys())
		image_paths = x[model_names[0]]["image"].unique()
		nms_res = {'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[], 'confidence':[],'name':[], 'image':[]}
		for img_path in image_paths:
			boxes = []
			scores = []
			for mod_name in model_names:
				preds = x[mod_name][x[mod_name]["image"]==img_path]
				for index, lab in preds.iterrows():
					boxes.append((
						lab['xmin'],				# x
						lab['ymin'],				# y
						lab['xmax'] - lab['xmin'],	# w
						lab['ymax'] - lab['ymin'],	# h
					))
					scores.append(lab['confidence'])
			indexes = cv2.dnn.NMSBoxes(boxes, scores,score_threshold=0.4,nms_threshold=0.8)
			for ind in indexes:
				i = ind
				if type(ind)==list:
					i = ind[0]
				file_name = img_path.split('/')[-1][0:-4]
				nms_res['xmin'] += [boxes[i][0]]
				nms_res['ymin'] += [boxes[i][1]]
				nms_res['xmax'] += [boxes[i][0] + boxes[i][2]]
				nms_res['ymax'] += [boxes[i][1] + boxes[i][3]]
				nms_res['confidence'] += [scores[i]]
				nms_res['name'] += [file_name]
				nms_res['image'] += [img_path]
		nms_res = pd.DataFrame(nms_res)
		print(nms_res)
		return nms_res

	def train(self, x, y) -> np.array:
		preds = self.predict(x)
		image_names_list = y["name"].unique()
		
		results = {
			'training_results': 0,
		}
		return results, preds


obj_det_input = pipeline_input("obj_det", 
	p_dataset_interpreter={
		# 'KITTI_lemenko_interp':KITTI_lemenko_interp,
		'karthika95-pedestrian-detection': obj_det_interp_1, 
	}, 
	p_model={
		'obj_det_pipeline_model_yolov5n': obj_det_pipeline_model_yolov5n,
		# 'obj_det_pipeline_model_yolov5s': obj_det_pipeline_model_yolov5s,
		# 'obj_det_pipeline_model_yolov5m': obj_det_pipeline_model_yolov5m,
		# 'obj_det_pipeline_model_yolov5l': obj_det_pipeline_model_yolov5l,
		# 'obj_det_pipeline_model_yolov5x': obj_det_pipeline_model_yolov5x,
	}, 
	p_ensembler={
		'obj_det_pipeline_ensembler_1': obj_det_pipeline_ensembler_1
	}, 
	p_vizualizer={
		'obj_det_data_visualizer': obj_det_data_visualizer
	})

#from depth_perception_demo import depth_input
exported_pipeline = obj_det_input
#all_inputs = {}
#all_inputs[obj_det_input.get_pipeline_name()] = obj_det_input
#all_inputs[depth_input.get_pipeline_name()] = depth_input


#########################################################################

def get_iou(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
		The (xmin, ymin) position is at the top left corner,
		the (xmax, ymax) position is at the bottom right corner
	bb2 : dict
		Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
		The (x, y) position is at the top left corner,
		the (xmax, ymax) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""
	assert bb1['xmin'] < bb1['xmax']
	assert bb1['ymin'] < bb1['ymax']
	assert bb2['xmin'] < bb2['xmax']
	assert bb2['ymin'] < bb2['ymax']

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1['xmin'], bb2['xmin'])
	y_top = max(bb1['ymin'], bb2['ymin'])
	x_right = min(bb1['xmax'], bb2['xmax'])
	y_bottom = min(bb1['ymax'], bb2['ymax'])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
	bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou