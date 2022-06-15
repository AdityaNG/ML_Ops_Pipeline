import os

RAW_DATASET_DIR = "data/{pipeline_name}/raw_datasets/"
DATASET_DIR = "data/{pipeline_name}/datasets/"

#MODEL_TRAINING_SETTINGS = "data/{pipeline_name}/model_training_settings.pkl"
#MODEL_VALIDATION_SETTINGS = "data/{pipeline_name}/model_validation_settings.pkl"

MODEL_TRAINING = "data/{pipeline_name}/models/{model_name}/training/{interpreter_name}"
MODEL_TESTING = "data/{pipeline_name}/models/{model_name}/testing/{interpreter_name}"

ENSEMBLE_TRAINING = "data/{pipeline_name}/ensemblers/{ensembler_name}/training/{interpreter_name}"
ENSEMBLE_TESTING = "data/{pipeline_name}/ensemblers/{ensembler_name}/testing/{interpreter_name}"

# Hight and width of the images
IMAGE_SIZE = 32
# 3 channels, Red, Green and Blue
CHANNELS = 3
# Number of epochs
#NUM_EPOCH = 350
NUM_EPOCH = 1
# learning rate
LEARN_RATE = 1.0e-4

SLEEP_TIME = 1

if __name__=="__main__":
	#from cifar10_demo import all_inputs
	from obj_det_demo import all_inputs
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

			ensemblers = all_inputs[pipeline_name].get_pipeline_ensembler()
			for ensembler_name in ensemblers:
				training_dir = ENSEMBLER_TRAINING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)
				os.makedirs(training_dir, exist_ok=True)
				print(training_dir)
				testing_dir = ENSEMBLER_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)
				os.makedirs(testing_dir, exist_ok=True)
				print(testing_dir)
				print('\t',ensembler_name, ensemblers[ensembler_name])
