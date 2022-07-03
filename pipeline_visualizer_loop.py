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

import time

from ensemble_visualizer import vizualize_ensemble
from constants import MODEL_TESTING, ENSEMBLE_TESTING, DATASET_DIR
from all_pipelines import get_all_inputs

def main():
	for pipeline_name in all_inputs:
		all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
		interpreters = all_inputs[pipeline_name].get_pipeline_dataset_interpreter()
		for interpreter_name in interpreters:
			interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
			interpreter_datasets = glob.glob(os.path.join(interpreter_dataset_dir,"*"))
			for dataset_dir in interpreter_datasets:
				dat = interpreters[interpreter_name](dataset_dir).get_dataset()
				ensemble_classes = all_inputs[pipeline_name].get_pipeline_ensembler()
				
				
				for ensembler_name in ensemble_classes:
					print("-"*10)
					print("ensembler_name:\t",ensembler_name)
					print("interpreter_name:\t",interpreter_name)
					print("dataset_dir:\t",dataset_dir)
					ens_testing_dir = ENSEMBLE_TESTING.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, ensembler_name=ensembler_name)
					dataset_name = dataset_dir.split("/")[-1]
					visualizers = list(all_inputs[pipeline_name].get_pipeline_visualizer().keys())
					for visualizer_name in visualizers:
						vizualize_ensemble(all_inputs[pipeline_name], 
							interpreter_name, dataset_name, ensembler_name, visualizer_name
						)

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
