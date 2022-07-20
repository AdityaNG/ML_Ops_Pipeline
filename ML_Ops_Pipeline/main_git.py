"""
model_analysis
"""

import torch

import glob
import pickle
import os
import time
import inspect
from datetime import datetime
#from multiprocessing import Pool
#import multiprocessing

import json

from .all_pipelines_git import get_all_inputs
from .pipeline_input import source_hash
from .constants import DATASET_DIR, ENSEMBLE_TRAINING, MODEL_TRAINING
from .history import local_history

import traceback

from .model_utils.model_training import train_model
from .model_utils.model_analysis import analyze_model

from .ensemble_utils.ensemble_training import train_ensemble
from .ensemble_utils.ensemble_analysis import analyze_ensemble
# from model_visualizer_loop import vi

def main(disable_torch_multiprocessing=False):
	loc_hist = local_history(__file__)
	task_list = {}
	all_inputs = get_all_inputs()

	for pipeline_name in all_inputs:
		all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		interpreters = all_inputs[pipeline_name]['pipeline'].get_pipeline_dataset_interpreter()
		for interpreter_name in interpreters:
			interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
			interpreter_datasets = glob.glob(os.path.join(interpreter_dataset_dir,"*"))
			for dataset_dir in interpreter_datasets:
				
				model_classes = all_inputs[pipeline_name]['pipeline'].get_pipeline_model()
				for model_name in model_classes:
					
					model_file_path = inspect.getfile(model_classes[model_name])
					#model_last_modified = str(datetime.fromtimestamp(os.path.getmtime(model_file_path)))
					#model_last_modified = str(source_hash(model_classes[model_name]))
					git_data = all_inputs[pipeline_name]['git_data']
					model_last_modified = git_data.hexsha
					task_id = model_name + ":"+ interpreter_name + ":" + dataset_dir
					
					if loc_hist[task_id] != model_last_modified:
						task_list.setdefault(pipeline_name, {})
						task_list[pipeline_name].setdefault(interpreter_name, {})
						task_list[pipeline_name][interpreter_name].setdefault(dataset_dir, {})
						task_list[pipeline_name][interpreter_name][dataset_dir].setdefault(model_name, (task_id, model_last_modified))

	if task_list == {}:
		print("Waiting for new tasks...")
		return

	print("-"*10)
	print("Task list:\n", json.dumps(task_list, sort_keys=True, indent=4))
	print("-"*10)

	pool_args = []
	pool_args_ensemble = []

	for pipeline_name in task_list:
		all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		interpreters = all_inputs[pipeline_name]['pipeline'].get_pipeline_dataset_interpreter()
		for interpreter_name in task_list[pipeline_name].keys():
			interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
			interpreter_datasets = task_list[pipeline_name][interpreter_name].keys()
			for dataset_dir in interpreter_datasets:
				
				#dat = interpreters[interpreter_name](dataset_dir).get_dataset()
				model_classes = all_inputs[pipeline_name]['pipeline'].get_pipeline_model()
				for model_name in task_list[pipeline_name][interpreter_name][dataset_dir].keys():
					print("-"*10)
					print("model_name:\t",model_name)
					print("interpreter_name:\t",interpreter_name)
					print("dataset_dir:\t",dataset_dir)
					task_id, model_last_modified = task_list[pipeline_name][interpreter_name][dataset_dir][model_name]
					training_dir = MODEL_TRAINING.format(
						pipeline_name=pipeline_name,
						interpreter_name=interpreter_name,
						model_name=model_name,
						commit_id=model_last_modified
					)
					os.makedirs(training_dir, exist_ok=True)
					if loc_hist[task_id] != model_last_modified:
						pool_args.append((pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified))
				
				ensemble_classes = all_inputs[pipeline_name]['pipeline'].get_pipeline_ensembler()
				for ensemble_name in ensemble_classes:
					print("-"*10)
					print("ensemble_name:\t",ensemble_name)
					print("interpreter_name:\t",interpreter_name)
					print("dataset_dir:\t",dataset_dir)
					task_id, model_last_modified = task_list[pipeline_name][interpreter_name][dataset_dir][model_name]
					task_id += ":" + ensemble_name
					ensemble_training_dir = ENSEMBLE_TRAINING.format(
						pipeline_name=pipeline_name,
						interpreter_name=interpreter_name,
						ensemble_name=ensemble_name,
						commit_id=model_last_modified
					)
					os.makedirs(ensemble_training_dir, exist_ok=True)
					if loc_hist[task_id] != model_last_modified:
						pool_args_ensemble.append((pipeline_name, ensemble_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified, ensemble_classes))

	if disable_torch_multiprocessing:
		for (pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified) in pool_args:
			status, task_id1, model_last_modified = train_model(pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified)
			status, task_id2, model_last_modified = analyze_model(pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified)
			loc_hist[task_id1] = model_last_modified
			loc_hist[task_id2] = model_last_modified

		for (pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified, ensemble_classes) in pool_args_ensemble:
			status, task_id1, model_last_modified = train_ensemble(pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified, ensemble_classes)
			status, task_id2, model_last_modified = analyze_ensemble(pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified, ensemble_classes)
			loc_hist[task_id1] = model_last_modified
			loc_hist[task_id2] = model_last_modified
	else:
		with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as p:
		#with torch.multiprocessing.Pool(1) as p:
			res1 = p.starmap(train_model, pool_args)
			res2 = p.starmap(analyze_model, pool_args)
			for status, task_id, model_last_modified in res1:
				loc_hist[task_id] = model_last_modified
			for status, task_id, model_last_modified in res2:
				loc_hist[task_id] = model_last_modified

			res1 = p.starmap(train_model, pool_args_ensemble)
			res2 = p.starmap(analyze_model, pool_args_ensemble)
			for status, task_id, model_last_modified in res1:
				loc_hist[task_id] = model_last_modified
			for status, task_id, model_last_modified in res2:
				loc_hist[task_id] = model_last_modified
		
		

if __name__ == "__main__":
	import argparse

	torch.multiprocessing.set_start_method('spawn')

	parser = argparse.ArgumentParser()
	parser.add_argument('--single', action='store_true', help='Run the loop only once')
	parser.add_argument('--disable-torch-multiprocessing', action='store_true', help='Disable multiprocessing')
	args = parser.parse_args()

	if args.single:
		main(disable_torch_multiprocessing=args.disable_torch_multiprocessing)
		exit()
		
	while True:
		try:
			main(disable_torch_multiprocessing=args.disable_torch_multiprocessing)
			time.sleep(5)
		except Exception as e:
			traceback.print_exc()
			print("Exception: {}".format(e))
			time.sleep(1)
