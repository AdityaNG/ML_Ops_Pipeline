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
import pickle
import datetime

from constants import DATASET_DIR, ENSEMBLE_TESTING, MODEL_TESTING, DATA_BASE_DIR, MODEL_BASE
from pipeline_input import pipeline_input

def vizualize_ensemble(p_input: pipeline_input, interpreter_name: str, dataset_name: str, ensemble_name: str, visualizer_name: str):
	assert isinstance(p_input, pipeline_input)
	pipeline_name = p_input.get_pipeline_name()
	all_dataset_interpreters = p_input.get_pipeline_dataset_interpreter()
	if interpreter_name not in all_dataset_interpreters.keys() or interpreter_name=='':
		print("Interpreter does not exist")
		print("List of available interpreters for pipeline=", pipeline_name)
		print("\n".join(all_dataset_interpreters.keys()))
		exit()
	
	dataset_interp = p_input.get_pipeline_dataset_interpreter_by_name(interpreter_name)
	all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		
	interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
	dataset_dir = os.path.join(interpreter_dataset_dir, dataset_name)
	print(dataset_dir)
	if not os.path.exists(dataset_dir) or dataset_name=='':
		print("Dataset does not exist")
		print("List of available datasets for interpreter=", interpreter_name)
		print("\n".join(os.listdir(interpreter_dataset_dir)))
		exit()
	
	dat = dataset_interp(dataset_dir).get_dataset()

	model_classes = p_input.get_pipeline_model()
	model_predictions = {}
	for model_name in model_classes:
		mod_testing_dir = MODEL_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name)
		os.makedirs(mod_testing_dir, exist_ok=True)
		results_pkl = os.path.join(mod_testing_dir, "results.pkl")
		predictions_pkl = os.path.join(mod_testing_dir, "predictions.pkl")
		results_handle = open(results_pkl, 'rb')
		results = pickle.load(results_handle)
		results_handle.close()
		predictions_handle = open(predictions_pkl, 'rb')
		predictions = pickle.load(predictions_handle)
		predictions_handle.close()
		model_predictions[model_name] = predictions

	ens_testing_dir = ENSEMBLE_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensemble_name)
	ensemble_classes = p_input.get_pipeline_ensembler()
	if not os.path.exists(ens_testing_dir) or ensemble_name=='' or ensemble_name not in ensemble_classes:
		print("Ensemble does not exist")
		print("List of available Ensemble models for pipeline=", pipeline_name)
		#print("\n".join(os.listdir(MODEL_BASE.format(pipeline_name=pipeline_name))))
		print("\n".join(ensemble_classes.keys()))
		exit()

	ens_results_pkl = os.path.join(ens_testing_dir, "results.pkl")
	ens_predictions_pkl = os.path.join(ens_testing_dir, "predictions.pkl")

	if not os.path.exists(results_pkl) or not os.path.exists(predictions_pkl):
		print("Analysis data for the given combination has not been generated yet")
		exit()

	visualizer_classes = p_input.get_pipeline_visualizer()
	if visualizer_name=='' or visualizer_name not in visualizer_classes:
		print("Vizualizer does not exist")
		print("List of available Ensemble Vizualizer for pipeline=", pipeline_name)
		print("\n".join(visualizer_classes.keys()))
		exit()
	visualizer = p_input.get_pipeline_visualizer_by_name(visualizer_name)()

	ens_results_handle = open(ens_results_pkl, 'rb')
	ens_results = pickle.load(ens_results_handle)
	ens_results_handle.close()

	ens_predictions_handle = open(ens_predictions_pkl, 'rb')
	ens_predictions = pickle.load(ens_predictions_handle)
	ens_predictions_handle.close()

	print("-"*os.get_terminal_size().columns)
	print("ensemble_classes:\t",ensemble_classes)
	print("interpreter_name:\t",interpreter_name)
	print("dataset_dir:\t",dataset_dir)

	print(results)
	print(predictions)

	visualizer.visualize(dat['test']['x'], dat['test']['y'], ens_predictions)

if __name__=="__main__":
	from obj_det_demo import all_inputs
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--pipeline_name', type=str, default='')
	parser.add_argument('--interpreter_name', type=str, default='')
	parser.add_argument('--dataset_name', type=str, default='')
	parser.add_argument('--ensemble_name', type=str, default='')
	parser.add_argument('--visualizer_name', type=str, default='')
	args = parser.parse_args()
	if args.pipeline_name not in all_inputs:
		print("Pipeline does not exist")
		print("List of available pipelines")
		print("\n".join(all_inputs.keys()))
		exit()

	vizualize_ensemble(all_inputs[args.pipeline_name], args.interpreter_name, args.dataset_name, args.ensemble_name, args.visualizer_name)
