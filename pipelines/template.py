
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import sys
sys.path.append('../')

from pipeline_input import pipeline_data_visualizer, pipeline_dataset_interpreter, pipeline_ensembler, pipeline_model, pipeline_input
from constants import *

class template_interp_1(pipeline_dataset_interpreter):
	def load(self) -> None:
		super().load()
		print("Loading from:", self.input_dir)
		# TODO
		self.dataset = {
			'train': {'x': [],'y': []},
			'test': {'x': [],'y': []}
		}

class template_data_visualizer(pipeline_data_visualizer):

	def visualize(self, x, y, preds, save_dir) -> None:
		# TODO: Visualize the data
		print(x)

class template_evaluator:

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


class template_pipeline_model(template_evaluator, pipeline_model):

	def load(self):
		# TODO: Load the model
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
		
	def train(self, x, y):
		preds = self.predict(x)
		# TODO: Train the model		
		results = {
			'training_results': 0,
		}
		return results, preds

	def predict(self, x: dict) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		predict_results = {
			'result': [], 'another_result': []
		}
		for i in tqdm(x):
			# TODO produce model predictions
			predict_results["result"] += ["Some Result"]
			predict_results["another_result"] += ["Another Result"]
				
		predict_results = pd.DataFrame(predict_results)
		return predict_results


class template_pipeline_ensembler_1(template_evaluator, pipeline_ensembler):

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


template_input = pipeline_input("template", 
	{
		'karthika95-pedestrian-detection': template_interp_1
	}, {
		'template_pipeline_model': template_pipeline_model,
	}, {
		'template_pipeline_ensembler_1': template_pipeline_ensembler_1
	}, {
		'template_data_visualizer': template_data_visualizer
	})

# Write the pipeline object to exported_pipeline
exported_pipeline = template_input
