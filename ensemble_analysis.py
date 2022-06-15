from xml.parsers.expat import model
from constants import *
import glob
import pickle
import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
import time

from helper import predict_classes, visualize_errors, plot_model

from obj_det_demo import all_inputs

def main():
	for pipeline_name in all_inputs:
		all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		interpreters = all_inputs[pipeline_name].get_pipeline_dataset_interpreter()
		for interpreter_name in interpreters:
			interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
			interpreter_datasets = glob.glob(os.path.join(interpreter_dataset_dir,"*"))
			for dataset_dir in interpreter_datasets:
				dat = interpreters[interpreter_name](dataset_dir).get_dataset()
				ensemble_classes = all_inputs[pipeline_name].get_pipeline_ensembler()
				
				model_classes = all_inputs[pipeline_name].get_pipeline_model()
				model_predictions = {}
				for model_name in model_classes:
					#print("-"*os.get_terminal_size().columns)
					#print("model_name:\t",model_name)
					#print("interpreter_name:\t",interpreter_name)
					#print("dataset_dir:\t",dataset_dir)
					testing_dir = MODEL_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name)
					os.makedirs(testing_dir, exist_ok=True)
					results_pkl = os.path.join(testing_dir, "results.pkl")
					predictions_pkl = os.path.join(testing_dir, "predictions.pkl")
					
					results_handle = open(results_pkl, 'rb')
					results = pickle.load(results_handle)
					results_handle.close

					predictions_handle = open(predictions_pkl, 'rb')
					predictions = pickle.load(predictions_handle)
					predictions_handle.close
					
					model_predictions[model_name] = predictions
				
				for ensembler_name in ensemble_classes:
					print("-"*os.get_terminal_size().columns)
					print("ensembler_name:\t",ensembler_name)
					print("interpreter_name:\t",interpreter_name)
					print("dataset_dir:\t",dataset_dir)
					testing_dir = ENSEMBLE_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)
					ens = ensemble_classes[ensembler_name]()
					#mod.predict(dat['test'])
					results = ens.evaluate(model_predictions, dat['test']['y'])
					print(results)
					print("-"*os.get_terminal_size().columns)

if __name__ == "__main__":
	import traceback
	while True:
		try:
			main()
		except Exception as e:
			traceback.print_exc()
			print("Exception: {}".format(e))
			time.sleep(1)
