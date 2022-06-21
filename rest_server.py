from flask import Flask, json, jsonify

import os
import glob
from constants import DATASET_DIR, DATA_BASE_DIR, MODEL_BASE, MODEL_TESTING, ENSEMBLE_BASE, ENSEMBLE_TESTING
import pickle
from flask_cors import CORS

api = Flask(__name__)
CORS(api)

@api.route('/pipelines', methods=['GET'])
def get_pipelines():
	# Method 1: Load the models and return the list
	# Issues: Single point of failiure
	# from obj_det_demo import all_inputs
	# pipeline_list = list(all_inputs.keys())
	# del all_inputs
	# Method 2: Generate list from data/ directory
	pipeline_list = os.listdir(DATA_BASE_DIR)
	return jsonify({'pipelines': pipeline_list})

@api.route('/datasets', methods=['GET'])
def get_datasets():
	pipeline_name = "obj_det"
	datasets_list = []
	all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
	interpreters = os.listdir(all_dataset_dir)
	for index, interpreter_name in enumerate(interpreters):
		interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
		interpreter_datasets = glob.glob(os.path.join(interpreter_dataset_dir,"*"))
		for dataset_dir in interpreter_datasets:
			time_stamp = dataset_dir.split("/")[-1]
			datasets_list.append({
				"indexNumber": index+1,
				"interpreter": interpreter_name,
				"size": sizeof_fmt(get_size(dataset_dir)),
				"issueDate": time_stamp,
				"name": interpreter_name
			})
	return jsonify({'datasets': datasets_list})

@api.route('/models', methods=['GET'])
def get_models():
	pipeline_name = "obj_det"
	#pipeline_name = "depth_det"
	models_list = []
	all_models_dir = MODEL_BASE.format(pipeline_name=pipeline_name)
	models = os.listdir(all_models_dir)
	for index, model_name in enumerate(models):
		model_dir = os.path.join(all_models_dir, model_name)
		time_stamp = model_dir.split("/")[-1]
		scores_list = []
		interpreters = os.listdir(os.path.join(model_dir, "testing"))
		for interpreter_name in interpreters:
			testing_dir = MODEL_TESTING.format(
				pipeline_name=pipeline_name,
				interpreter_name=interpreter_name,
				model_name=model_name
			)
			results_pkl = os.path.join(testing_dir, "results.pkl")
			try:
				results_handle = open(results_pkl, 'rb')
				results = pickle.load(results_handle)
				results_handle.close()

				scores_list.append({
					"interpreter": interpreter_name,
					"results": results
				})
			except Exception as ex:
				print(ex)
				scores_list.append({
					"interpreter": interpreter_name,
					"results": None,
					"error": ex
				})
				
		models_list.append({
			"indexNumber": index+1,
			"model": model_name,
			"size": sizeof_fmt(get_size(model_dir)),
			"issueDate": time_stamp,
			"name": model_name,
			"scores_list": scores_list,
			"status": "Done"
		})
	return jsonify({'models': models_list})


@api.route('/ensembles', methods=['GET'])
def get_ensembles():
	pipeline_name = "obj_det"
	#pipeline_name = "depth_det"
	ensembles_list = []
	all_ensembles_dir = ENSEMBLE_BASE.format(pipeline_name=pipeline_name)
	ensembles = os.listdir(all_ensembles_dir)
	for index, ensemble_name in enumerate(ensembles):
		ensemble_dir = os.path.join(all_ensembles_dir, ensemble_name)
		time_stamp = ensemble_dir.split("/")[-1]
		scores_list = []
		interpreters = os.listdir(os.path.join(ensemble_dir, "testing"))
		for interpreter_name in interpreters:
			testing_dir = ENSEMBLE_TESTING.format(
				pipeline_name=pipeline_name,
				interpreter_name=interpreter_name,
				ensembler_name=ensemble_name
			)
			results_pkl = os.path.join(testing_dir, "results.pkl")
			try:
				results_handle = open(results_pkl, 'rb')
				results = pickle.load(results_handle)
				results_handle.close()

				scores_list.append({
					"interpreter": interpreter_name,
					"results": results
				})
			except Exception as ex:
				print(ex)
				scores_list.append({
					"interpreter": interpreter_name,
					"results": None,
					"error": ex
				})
				
		ensembles_list.append({
			"indexNumber": index+1,
			"ensemble": ensemble_name,
			"size": sizeof_fmt(get_size(ensemble_dir)),
			"issueDate": time_stamp,
			"name": ensemble_name,
			"scores_list": scores_list,
			"status": "Done"
		})
	return jsonify({'ensembles': ensembles_list})

##################################

def get_size(start_path):
	total_size = 0
	for dirpath, dirnames, filenames in os.walk(start_path):
		for f in filenames:
			fp = os.path.join(dirpath, f)
			# skip if it is symbolic link
			if not os.path.islink(fp):
				total_size += os.path.getsize(fp)

	return total_size


def sizeof_fmt(num, suffix="B"):
	for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
		if abs(num) < 1024.0:
			return f"{num:3.1f} {unit}{suffix}"
		num /= 1024.0
	return f"{num:.1f}Yi{suffix}"


if __name__ == '__main__':
	api.run() 
