"""
ensemble analysis

Runs analysis on all the enseble models.
Loads in the datasets and outputs from the individual models.
Runs the inputs through all the enseble models and verifies outputs
against the dataset ground truth
"""

import glob
import pickle
import os
import inspect
from datetime import datetime
import time

import json

from constants import MODEL_TESTING, ENSEMBLE_TESTING, DATASET_DIR
from all_pipelines import get_all_inputs
from history import local_history

def main():
	loc_hist = local_history(__file__)
	task_list = {}
	all_inputs = get_all_inputs()

	for pipeline_name in all_inputs:
		all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		interpreters = all_inputs[pipeline_name].get_pipeline_dataset_interpreter()
		for interpreter_name in interpreters:
			interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
			interpreter_datasets = glob.glob(os.path.join(interpreter_dataset_dir,"*"))
			for dataset_dir in interpreter_datasets:
				ensemble_classes = all_inputs[pipeline_name].get_pipeline_ensembler()
				
				model_classes = all_inputs[pipeline_name].get_pipeline_model()
				model_predictions = {}
				
				model_did_change = False
				for model_name in model_classes:
					model_file_path = inspect.getfile(model_classes[model_name])
					model_last_modified = datetime.fromtimestamp(os.path.getmtime(model_file_path))
					model_task_id = "MODEL_" + model_name + ":"+ interpreter_name + ":" + dataset_dir
					if loc_hist[model_task_id] != model_last_modified:
						model_did_change = True
						break
				
				if model_did_change:
					for ensembler_name in ensemble_classes:
						ensemble_file_path = inspect.getfile(ensemble_classes[ensembler_name])
						ensemble_last_modified = datetime.fromtimestamp(os.path.getmtime(ensemble_file_path))
						task_id = model_name + ":"+ interpreter_name + ":" + dataset_dir
						if loc_hist[task_id] != ensemble_last_modified:
							task_list.setdefault(pipeline_name, {})
							task_list[pipeline_name].setdefault(interpreter_name, {})
							task_list[pipeline_name][interpreter_name].setdefault(dataset_dir, {})
							task_list[pipeline_name][interpreter_name][dataset_dir].setdefault(ensembler_name, task_id)

	if task_list == {}:
		print("Waiting for new tasks...")
		time.sleep(5)
		return

	print("-"*10)
	print("Task list:\n", json.dumps(task_list, sort_keys=True, indent=4))
	print("-"*10)

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

					testing_dir = MODEL_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name)
					os.makedirs(testing_dir, exist_ok=True)
					results_pkl = os.path.join(testing_dir, "results.pkl")
					predictions_pkl = os.path.join(testing_dir, "predictions.pkl")
					
					results_handle = open(results_pkl, 'rb')
					results = pickle.load(results_handle)
					results_handle.close()

					predictions_handle = open(predictions_pkl, 'rb')
					predictions = pickle.load(predictions_handle)
					predictions_handle.close()
					
					model_predictions[model_name] = predictions

				for ensembler_name in ensemble_classes:
					
					print("-"*10)
					print("ensembler_name:\t",ensembler_name)
					print("interpreter_name:\t",interpreter_name)
					print("dataset_dir:\t",dataset_dir)
					ens_testing_dir = ENSEMBLE_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)
					ens = ensemble_classes[ensembler_name]()
					#mod.predict(dat['test'])
					ens_results, ens_predictions = ens.evaluate(model_predictions, dat['test']['y'])
					print(results)
					
					ens_results_pkl = os.path.join(ens_testing_dir, "results.pkl")
					ens_predictions_pkl = os.path.join(ens_testing_dir, "predictions.pkl")

					ens_results_handle = open(ens_results_pkl, 'wb')
					pickle.dump(ens_results, ens_results_handle, protocol=pickle.HIGHEST_PROTOCOL)
					ens_results_handle.close()

					ens_predictions_handle = open(ens_predictions_pkl, 'wb')
					pickle.dump(ens_predictions, ens_predictions_handle, protocol=pickle.HIGHEST_PROTOCOL)
					ens_predictions_handle.close()

					task_id = task_list[pipeline_name][interpreter_name][dataset_dir][ensembler_name]
					loc_hist[task_id] = model_last_modified

if __name__ == "__main__":
	import traceback
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--single', action='store_true', help='Run the loop only once')
	args = parser.parse_args()

	if args.single:
		main()
		exit()

	while True:
		try:
			main()
		except Exception as ex:
			print("-"*10)
			print("An error has occured")
			print(ex)
			traceback.print_exc()
			print("-"*10)
			time.sleep(1)
