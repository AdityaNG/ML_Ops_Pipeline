import os

if 'PIPELINE_HOME' in os.environ:
	PIPELINE_HOME = os.environ['PIPELINE_HOME']
	assert os.path.exists(PIPELINE_HOME)
else:
	PIPELINE_HOME = os.path.join(os.path.expanduser('~'), ".ML_Ops_Pipeline")

DATA_BASE_DIR = os.path.join(PIPELINE_HOME, "data/")
RAW_DATASET_DIR = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/raw_datasets/")
DATASET_DIR = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/datasets/")
LOG_DIR = os.path.join(PIPELINE_HOME, "logs/")
MLFLOW_DIR = os.path.join(PIPELINE_HOME, "mlflow/")
os.makedirs(LOG_DIR, exist_ok=True)
#MODEL_TRAINING_SETTINGS = "data/{pipeline_name}/model_training_settings.pkl"
#MODEL_VALIDATION_SETTINGS = "data/{pipeline_name}/model_validation_settings.pkl"

MODEL_BASE = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/models/")
MODEL_TRAINING = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/models/{model_name}/{commit_id}/training/{interpreter_name}")
MODEL_TESTING = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/models/{model_name}/{commit_id}/testing/{interpreter_name}")
MODEL_VISUAL = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/visuals/{model_name}/{commit_id}/{interpreter_name}/{visualizer_name}/")

ENSEMBLE_BASE = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/ensembles/")
ENSEMBLE_TRAINING = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/ensembles/{ensemble_name}/{commit_id}/training/{interpreter_name}")
ENSEMBLE_TESTING = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/ensembles/{ensemble_name}/{commit_id}/testing/{interpreter_name}")
ENSEMBLE_VISUAL = os.path.join(PIPELINE_HOME, "data/{pipeline_name}/visuals/{ensemble_name}/{commit_id}/{interpreter_name}/{visualizer_name}/")

HISTORY_PATH = os.path.join(PIPELINE_HOME, "history")

REMOTE_PIPELINES_DIR = os.path.join(PIPELINE_HOME, "remote_pipelines")
REMOTE_PIPELINES_TXT = os.path.join(PIPELINE_HOME, "remote_pipelines.txt")

def folder_last_modified(folder):
	last_modified = []
	for path, directories, files in os.walk(folder):
		for file in files:
			last_modified.append(os.path.getmtime(os.path.join(path, file)))
	#model_results_last_modified = max(os.path.getmtime(inspect.getfile(f)) for f in os.walk(testing_dir))
	if len(last_modified):
		return max(last_modified)
	return 0

def generate_tree():
	from all_pipelines import get_all_inputs
	os.makedirs(PIPELINE_HOME)
	all_inputs = get_all_inputs()
	for pipeline_name in all_inputs:
		raw_dataset_dir = RAW_DATASET_DIR.format(pipeline_name=pipeline_name)
		os.makedirs(raw_dataset_dir, exist_ok=True)
		print(raw_dataset_dir)
		dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		os.makedirs(dataset_dir, exist_ok=True)
		print(dataset_dir)
		print(pipeline_name, all_inputs[pipeline_name])
		interpreters = all_inputs[pipeline_name].get_pipeline_dataset_interpreter()
		for interpreter_name in interpreters:
			print('\t',interpreter_name, interpreters[interpreter_name])
			models = all_inputs[pipeline_name].get_pipeline_model()
			for model_name in models:
				training_dir = MODEL_TRAINING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name)
				os.makedirs(training_dir, exist_ok=True)
				print(training_dir)
				testing_dir = MODEL_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name)
				os.makedirs(testing_dir, exist_ok=True)
				print(testing_dir)
				print('\t',model_name, models[model_name])

			# ensemblers = all_inputs[pipeline_name].get_pipeline_ensemble()
			# for ensembler_name in ensemblers:
			# 	training_dir = ENSEMBLE_TRAINING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)
			# 	os.makedirs(training_dir, exist_ok=True)
			# 	print(training_dir)
			# 	testing_dir = ENSEMBLE_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)
			# 	os.makedirs(testing_dir, exist_ok=True)
			# 	print(testing_dir)
			# 	print('\t',ensembler_name, ensemblers[ensembler_name])

if __name__=="__main__":
	generate_tree()
