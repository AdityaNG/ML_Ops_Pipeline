"""
data_ingestion

Takes a path as input along with pipeline name and the interpreter name
Reads the dataset using the said pipeline's specified interpreter
If the interpreter raises no errors, the path is copied into datasets
If any errrors occur or assertions fail, path is rejected
"""
from distutils.dir_util import copy_tree
import traceback
import os
import datetime

from constants import DATASET_DIR
from pipeline_input import pipeline_input

def ingest_data(p_input: pipeline_input, INTERPRETER_NAME: str, input_dir: str):
	assert isinstance(p_input, pipeline_input)
	dataset_interp_class = p_input.get_pipeline_dataset_interpreter_by_name(INTERPRETER_NAME)
	pipeline_name = p_input.get_pipeline_name()
	PIPELINE_BASE_FOLDER = DATASET_DIR.format(pipeline_name=pipeline_name)
	os.makedirs(PIPELINE_BASE_FOLDER, exist_ok=True)
	BASE_FOLDER_ID = os.path.join(PIPELINE_BASE_FOLDER, INTERPRETER_NAME, str(datetime.datetime.now()).replace(" ", "_"))
	try:
		dataset_interp_class(input_dir)
		print("Interpreter accepted, copying...")
		print(input_dir, '->',BASE_FOLDER_ID)
		os.makedirs(BASE_FOLDER_ID, exist_ok=True)
		copy_tree(input_dir, BASE_FOLDER_ID)
	except AssertionError as ex:
		print("Interpreter rejected, aborting...")
		print(ex)
		traceback.print_exc()


if __name__=="__main__":
	from obj_det_demo import all_inputs
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--pipeline_name', type=str, required=True)
	parser.add_argument('--interpreter_name', type=str, required=True)
	args = parser.parse_args()
	ingest_data(all_inputs[args.pipeline_name], args.interpreter_name, args.input_dir)
