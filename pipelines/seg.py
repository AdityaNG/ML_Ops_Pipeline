
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import glob

from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')

from pipeline_input import pipeline_data_visualizer, pipeline_dataset_interpreter, pipeline_ensembler, pipeline_model, pipeline_input
from constants import *

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



class seg_kitti(pipeline_dataset_interpreter):

	def generate_data(self, dataset):
		dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=1)
		x_train, y_train = dataset_train[['image_2']], dataset_train[['instance', 'semantic', 'semantic_rgb']]
		x_test, y_test = dataset_test[['image_2']], dataset_test[['instance', 'semantic', 'semantic_rgb']]
		return x_train, y_train, x_test, y_test


	def load(self) -> None:

		dataset = {}

		for mode in ('testing', 'training'):
			print("Loading seg_kitti from:", self.input_dir)
			image_2_folder = os.path.join(self.input_dir, mode, "image_2")
			instance_folder = os.path.join(self.input_dir, mode, "instance")
			semantic_folder = os.path.join(self.input_dir, mode, "semantic")
			semantic_rgb_folder = os.path.join(self.input_dir, mode, "semantic_rgb")

			assert os.path.exists(image_2_folder), image_2_folder
			if mode=='training':
				assert os.path.exists(instance_folder), instance_folder
				assert os.path.exists(semantic_folder), semantic_folder
				assert os.path.exists(semantic_rgb_folder), semantic_rgb_folder

			if mode=='training':
				dataset[mode] = {
					'image_2': [], 'instance': [], 'semantic': [], 'semantic_rgb': []
				}
			else:
				dataset[mode] = {
					'image_2': []
				}
			image_2_files_list = sorted(glob.glob(os.path.join(image_2_folder, "*.png")))
			files_list = list(map(lambda x: x.split("/")[-1].split(".png")[:-1][0], image_2_files_list))
			#print(files_list)
			print("seg_kitti: load", mode)
			for f in tqdm(files_list):
				image_2_path = os.path.join(image_2_folder, f+".png")
				instance_path = os.path.join(instance_folder, f+".png")
				semantic_path = os.path.join(semantic_folder, f+".png")
				semantic_rgb_path = os.path.join(semantic_rgb_folder, f+".png")
					
				assert os.path.exists(image_2_path), image_2_path
				if mode=="training":
					assert os.path.exists(instance_path), instance_path
					assert os.path.exists(semantic_path), semantic_path
					assert os.path.exists(semantic_rgb_path), semantic_rgb_path
				
				dataset[mode]['image_2'] += [image_2_path]
				if mode=='training':
					dataset[mode]['instance'] += [instance_path]
					dataset[mode]['semantic'] += [semantic_path]
					dataset[mode]['semantic_rgb'] += [semantic_rgb_path]
				
			dataset[mode] = pd.DataFrame(dataset[mode])
		
		# Discard the KITTI test files, because no ground truth
		x_train, y_train, x_test, y_test = self.generate_data(dataset['training'])
		self.dataset = {
			'train': {
				'x': x_train,
				'y': y_train
			},
			'test': {
				'x': x_test,
				'y': y_test
			}
		}

class seg_data_visualizer(pipeline_data_visualizer):

	def visualize(self, x, y, preds, save_dir) -> None:
		# TODO: Visualize the data
		print(x)

class seg_evaluator:

	def evaluate(self, x, y):
		preds = self.predict(x)
		# TODO: write a common evaluation script that is common to all models
		# Note: Optionally, you can give model specific implementations for the evaluation logic
		#		by overloading the evaluate(self, x, y) method in the model class
		results = {
			'some_metric': 0,
			'another_metric': 0
		}
		return results, preds


class seg_pipeline_model(seg_evaluator, pipeline_model):

	def load(self):
		cfg = get_cfg()
		# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
		cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
		# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
		cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		self.predictor = DefaultPredictor(cfg)
		
		pass
		
	def train(self, x, y) -> np.array:
		preds = self.predict(x)
		# TODO: Train the model		
		results = {
			'training_results': 0,
		}
		return results, preds

	def predict(self, x) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		predict_results = {
			'result': [], 'another_result': []
		}
		for i in tqdm(x):
			# TODO produce model predictions
			outputs = self.predictor(im)
			predict_results["result"] += ["Some Result"]
			predict_results["another_result"] += ["Another Result"]
				
		predict_results = pd.DataFrame(predict_results)
		return predict_results


class seg_pipeline_ensembler_1(seg_evaluator, pipeline_ensembler):

	def predict(self, x: dict) -> np.array:
		model_names = list(x.keys())
		predict_results = {
			'result': [], 'another_result': []
		}
		for i in tqdm(x):
			for mod_name in model_names:
				preds = x[mod_name]
			# TODO produce ensebled results based on all the model's predictions
			predict_results["result"] += ["Some Ensembled Result"]
			predict_results["another_result"] += ["Another Ensembled Result"]
				
		predict_results = pd.DataFrame(predict_results)
		return predict_results


	def train(self, x, y) -> np.array:
		preds = self.predict(x)
		# TODO: train ensemble model
		results = {
			'training_results': 0,
		}
		return results, preds


seg_input = pipeline_input("seg", 
	{
		'seg_kitti': seg_kitti
	}, {
		'seg_pipeline_model': seg_pipeline_model,
	}, {
		'seg_pipeline_ensembler_1': seg_pipeline_ensembler_1
	}, {
		'seg_data_visualizer': seg_data_visualizer
	})

# Write the pipeline object to exported_pipeline
exported_pipeline = seg_input
