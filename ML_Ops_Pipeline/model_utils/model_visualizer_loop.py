"""
model_analysis
"""

import glob
import pickle
import os
import time
import inspect
from datetime import datetime
import traceback

import json
from typing import final

from sklearn import pipeline

from ..all_pipelines_git import get_all_inputs
from ..pipeline_input import source_hash
from ..constants import DATASET_DIR, MODEL_TESTING, MODEL_TRAINING, MODEL_VISUAL, folder_last_modified
from ..history import local_history

def visualize_model(pipeline_name, model_name, interpreter_name, dataset_dir, task_id, model_last_modified, visualizers, visualizer_name, dat, mode):
	print("-"*10)
	print("model_name:\t",model_name)
	print("interpreter_name:\t",interpreter_name)
	print("dataset_dir:\t",dataset_dir)

	testing_dir = MODEL_TESTING.format(
		pipeline_name=pipeline_name,
		interpreter_name=interpreter_name,
		model_name=model_name,
		commit_id=model_last_modified
	)
	os.makedirs(testing_dir, exist_ok=True)
	training_dir = MODEL_TRAINING.format(
		pipeline_name=pipeline_name,
		interpreter_name=interpreter_name,
		model_name=model_name,
		commit_id=model_last_modified
	)
	os.makedirs(training_dir, exist_ok=True)

	if mode=='train':
		results_pkl = os.path.join(training_dir, "results.pkl")
		predictions_pkl = os.path.join(training_dir, "predictions.pkl")
	elif mode=='test':
		results_pkl = os.path.join(testing_dir, "results.pkl")
		predictions_pkl = os.path.join(testing_dir, "predictions.pkl")
	else:
		raise Exception("Mode must be test or train")

	results_handle = open(results_pkl, 'rb')
	results = pickle.load(results_handle)
	results_handle.close()
	predictions_handle = open(predictions_pkl, 'rb')
	predictions = pickle.load(predictions_handle)
	predictions_handle.close()

	visual_dir = MODEL_VISUAL.format(
		pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name, visualizer_name=visualizer_name,
		commit_id=model_last_modified
	)
	os.makedirs(visual_dir, exist_ok=True)

	visual_files = glob.glob(os.path.join(visual_dir, "*"))
	for vf in visual_files:
		os.remove(vf)
	os.makedirs(visual_dir, exist_ok=True)

	print("-"*10)
	print("model_name:\t",model_name)
	print("interpreter_name:\t",interpreter_name)
	print("dataset_dir:\t",dataset_dir)
	print("visual_dir:\t",visual_dir)

	visualizers[visualizer_name]().visualize(dat[mode]['x'], dat[mode]['y'], results, predictions, visual_dir)

	return (True, task_id, model_last_modified, visual_dir)
	
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
				
				model_classes = all_inputs[pipeline_name].get_pipeline_model()
				for model_name in model_classes:

					testing_dir = MODEL_TESTING.format(
						pipeline_name=pipeline_name,
						interpreter_name=interpreter_name,
						model_name=model_name
					)
					
					model_results_last_modified = str(datetime.fromtimestamp(folder_last_modified(testing_dir)))

					visualizers = all_inputs[pipeline_name].get_pipeline_visualizer()
					for visualizer_name in visualizers:
						visualizer_last_modified = str(source_hash(visualizers[visualizer_name]))

						visual_dir = MODEL_VISUAL.format(
							pipeline_name=pipeline_name, 
							interpreter_name=interpreter_name, 
							model_name=model_name, 
							visualizer_name=visualizer_name,
							commit_id=visualizer_last_modified
						)
						visual_dir_last_modified = str(datetime.fromtimestamp(folder_last_modified(visual_dir)))

						task_id = model_name + ":"+ interpreter_name + ":" + dataset_dir + ":" + visualizer_name
						task_last_modified = model_results_last_modified + visualizer_last_modified

						if loc_hist[task_id] != task_last_modified+visual_dir_last_modified:
							task_list.setdefault(pipeline_name, {})
							task_list[pipeline_name].setdefault(interpreter_name, {})
							task_list[pipeline_name][interpreter_name].setdefault(dataset_dir, {})
							task_list[pipeline_name][interpreter_name][dataset_dir].setdefault(model_name, {})
							task_list[pipeline_name][interpreter_name][dataset_dir][model_name].setdefault(visualizer_name, (task_id, task_last_modified, visual_dir_last_modified))

	if task_list == {}:
		#print("Waiting for new tasks...")
		return

	print("-"*10)
	print("Task list:\n", json.dumps(task_list, sort_keys=True, indent=4))
	print("-"*10)

	for pipeline_name in task_list:
		all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		interpreters = all_inputs[pipeline_name].get_pipeline_dataset_interpreter()
		for interpreter_name in task_list[pipeline_name].keys():
			interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
			interpreter_datasets = task_list[pipeline_name][interpreter_name].keys()
			for dataset_dir in interpreter_datasets:
				
				dat = interpreters[interpreter_name](dataset_dir).get_dataset()
				model_classes = all_inputs[pipeline_name].get_pipeline_model()
				for model_name in task_list[pipeline_name][interpreter_name][dataset_dir].keys():
					print("-"*10)
					print("model_name:\t",model_name)
					print("interpreter_name:\t",interpreter_name)
					print("dataset_dir:\t",dataset_dir)
					testing_dir = MODEL_TESTING.format(
						pipeline_name=pipeline_name,
						interpreter_name=interpreter_name,
						model_name=model_name
					)
					os.makedirs(testing_dir, exist_ok=True)
					dataset_name = dataset_dir.split("/")[-1]
					#visualizers = list(all_inputs[pipeline_name].get_pipeline_visualizer().keys())
					visualizers = all_inputs[pipeline_name].get_pipeline_visualizer()
					for visualizer_name in task_list[pipeline_name][interpreter_name][dataset_dir][model_name].keys():
						results_pkl = os.path.join(testing_dir, "results.pkl")
						predictions_pkl = os.path.join(testing_dir, "predictions.pkl")

						results_handle = open(results_pkl, 'rb')
						results = pickle.load(results_handle)
						results_handle.close()

						predictions_handle = open(predictions_pkl, 'rb')
						predictions = pickle.load(predictions_handle)
						predictions_handle.close()

						visual_dir = MODEL_VISUAL.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name, visualizer_name=visualizer_name)
						os.makedirs(visual_dir, exist_ok=True)

						visual_files = glob.glob(os.path.join(visual_dir, "*"))
						for vf in visual_files:
							os.remove(vf)
						os.makedirs(visual_dir, exist_ok=True)

						print("-"*10)
						print("model_name:\t",model_name)
						print("interpreter_name:\t",interpreter_name)
						print("dataset_dir:\t",dataset_dir)
						print("visual_dir:\t",visual_dir)

						try:
							visualizers[visualizer_name]().visualize(dat['test']['x'], dat['test']['y'], results, predictions, visual_dir)
						except Exception as ex:
							if isinstance(ex, KeyboardInterrupt): exit()
							print(ex)
							traceback.print_exc()
						finally:
							visual_dir_last_modified = str(datetime.fromtimestamp(folder_last_modified(visual_dir)))
							task_id, task_last_modified, visual_dir_last_modified_old = task_list[pipeline_name][interpreter_name][dataset_dir][model_name][visualizer_name]
							loc_hist[task_id] = task_last_modified + visual_dir_last_modified


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
			time.sleep(5)
		except Exception as e:
			traceback.print_exc()
			print("Exception: {}".format(e))
			time.sleep(1)
