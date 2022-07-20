"""
model_training
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

from ..all_pipelines import get_all_inputs
from ..pipeline_input import source_hash
from ..constants import DATASET_DIR, MODEL_TRAINING
from ..history import local_history

import traceback

def train_model(pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified):
	print("-"*10)
	print("model_name:\t",model_name)
	print("interpreter_name:\t",interpreter_name)
	print("dataset_dir:\t",dataset_dir)
	training_dir = MODEL_TRAINING.format(
		pipeline_name=pipeline_name,
		interpreter_name=interpreter_name,
		model_name=model_name,
		commit_id=model_last_modified
	)
	os.makedirs(training_dir, exist_ok=True)
	tb = "OK"
	try:
		dat = interpreters[interpreter_name](dataset_dir).get_dataset()
		mod = model_classes[model_name](training_dir)
		#mod.predict(dat['train'])
		results, predictions = mod.train(dat['train']['x'], dat['train']['y'])
		#print(results)

		results_pkl = os.path.join(training_dir, "results.pkl")
		predictions_pkl = os.path.join(training_dir, "predictions.pkl")
		model_pkl = os.path.join(training_dir, "model.pkl")

		results_handle = open(results_pkl, 'wb')
		pickle.dump(results, results_handle, protocol=pickle.HIGHEST_PROTOCOL)
		results_handle.close()

		predictions_handle = open(predictions_pkl, 'wb')
		pickle.dump(predictions, predictions_handle, protocol=pickle.HIGHEST_PROTOCOL)
		predictions_handle.close()

		model_handle = open(model_pkl, 'wb')
		pickle.dump(mod, model_handle, protocol=pickle.HIGHEST_PROTOCOL)
		model_handle.close()

		return (True, task_id, model_last_modified)
	except Exception as ex:
		print(ex)
		tb = traceback.format_exc()
	finally:
		print(tb)
		err_txt = os.path.join(training_dir, "err.txt")
		err_file = open(err_txt, "w")
		err_file.write(tb)
		err_file.close()
		return (False, task_id, model_last_modified)

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
					
					model_file_path = inspect.getfile(model_classes[model_name])
					#model_last_modified = str(datetime.fromtimestamp(os.path.getmtime(model_file_path)))
					model_last_modified = str(source_hash(model_classes[model_name]))
					task_id = model_name + ":"+ interpreter_name + ":" + dataset_dir
					
					if loc_hist[task_id] != model_last_modified:
						task_list.setdefault(pipeline_name, {})
						task_list[pipeline_name].setdefault(interpreter_name, {})
						task_list[pipeline_name][interpreter_name].setdefault(dataset_dir, {})
						task_list[pipeline_name][interpreter_name][dataset_dir].setdefault(model_name, (task_id, model_last_modified))

	if task_list == {}:
		#print("Waiting for new tasks...")
		return

	print("-"*10)
	print("Task list:\n", json.dumps(task_list, sort_keys=True, indent=4))
	print("-"*10)

	pool_args = []

	for pipeline_name in task_list:
		all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		interpreters = all_inputs[pipeline_name].get_pipeline_dataset_interpreter()
		for interpreter_name in task_list[pipeline_name].keys():
			interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
			interpreter_datasets = task_list[pipeline_name][interpreter_name].keys()
			for dataset_dir in interpreter_datasets:
				
				#dat = interpreters[interpreter_name](dataset_dir).get_dataset()
				model_classes = all_inputs[pipeline_name].get_pipeline_model()
				for model_name in task_list[pipeline_name][interpreter_name][dataset_dir].keys():
					print("-"*10)
					print("model_name:\t",model_name)
					print("interpreter_name:\t",interpreter_name)
					print("dataset_dir:\t",dataset_dir)
					training_dir = MODEL_TRAINING.format(
						pipeline_name=pipeline_name,
						interpreter_name=interpreter_name,
						model_name=model_name
					)
					os.makedirs(training_dir, exist_ok=True)
					
					task_id, model_last_modified = task_list[pipeline_name][interpreter_name][dataset_dir][model_name]
					
					pool_args.append((pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified))

	#with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as p:
	with torch.multiprocessing.Pool(1) as p:
		res = p.starmap(train_model, pool_args)
		for status, task_id, model_last_modified in res:
			#status, task_id, model_last_modified = train_model(pipeline_name, model_name, interpreter_name, dataset_dir, model_classes, interpreters, task_id, model_last_modified)
			if status:
				loc_hist[task_id] = model_last_modified

if __name__ == "__main__":
	import argparse

	torch.multiprocessing.set_start_method('spawn')# good solution !!!!

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
