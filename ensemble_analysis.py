from xml.parsers.expat import model
from constants import *
import glob
import pickle

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
		dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		interpreters = all_inputs[pipeline_name].get_pipeline_dataset_interpreter()
		for interpreter_name in interpreters:
			models = all_inputs[pipeline_name].get_pipeline_model()
			for model_name in models:
				training_dir = MODEL_TRAINING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name)
				testing_dir = MODEL_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name)

			ensemblers = all_inputs[pipeline_name].get_pipeline_ensembler()
			for ensembler_name in ensemblers:
				training_dir = ENSEMBLER_TRAINING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)
				testing_dir = ENSEMBLER_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)

if __name__ == "__main__":
	while True:
		try:
			main()
		except Exception as e:
			print("Exception: {}".format(e))
			time.sleep(1)
