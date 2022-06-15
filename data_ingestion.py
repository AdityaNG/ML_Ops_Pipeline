from distutils.dir_util import copy_tree
import shutil
import traceback
import os

from sklearn import datasets
from constants import *
from tqdm import tqdm
import cv2
import datetime
import pickle
import numpy as np
from keras.utils import np_utils
import glob

from pipeline_input import *

def ingest_data(p_input: pipeline_input, interpreter_name: str, input_dir: str):
	assert isinstance(p_input, pipeline_input)
	dataset_interp_class = p_input.get_pipeline_dataset_interpreter_by_name(interpreter_name)
	pipeline_name = p_input.get_pipeline_name()
	PIPELINE_BASE_FOLDER = DATASET_DIR.format(pipeline_name=pipeline_name)
	os.makedirs(PIPELINE_BASE_FOLDER, exist_ok=True)
	BASE_FOLDER_ID = os.path.join(PIPELINE_BASE_FOLDER, interpreter_name, str(datetime.datetime.now()).replace(" ", "_"))
	try:
		dat = dataset_interp_class(input_dir)
		print("Interpreter accepted, copying...")
		print(input_dir, '->',BASE_FOLDER_ID)
		os.makedirs(BASE_FOLDER_ID, exist_ok=True)
		copy_tree(input_dir, BASE_FOLDER_ID)

	except Exception as e:
		print("Interpreter rejected, aborting...")
		print(e)  
		traceback.print_exc()


if __name__=="__main__":
	# python3 data_ingestion.py --input_dir /home/lxd1kor/Downloads/karthika95-pedestrian-detection/ --pipeline_name obj_det --interpreter_name karthika95-pedestrian-detection 
	from obj_det_demo import all_inputs
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--pipeline_name', type=str, required=True)
	parser.add_argument('--interpreter_name', type=str, required=True)
	args = parser.parse_args()
	ingest_data(all_inputs[args.pipeline_name], args.interpreter_name, args.input_dir)
