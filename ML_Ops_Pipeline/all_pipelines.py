from cmath import pi
import glob
import os
import importlib
import traceback

log = False
if __name__=="__main__":
    log = True

global all_inputs, all_inputs_modules
all_inputs = {}
all_inputs_modules = {}

def get_all_inputs():
	global all_inputs, all_inputs_modules

	pipelines_list = list(map(lambda x: x.split("/")[-1],glob.glob('pipelines/*')))
	#pipelines_list = list(map(lambda x: x.split("/")[-1],glob.glob('remote_pipelines/*')))
	pipelines_list = list(filter(lambda x: x not in ["template", "__pycache__", "README.md"], pipelines_list))
	if log:
		print(pipelines_list)
		print("-"*10)

	for p in pipelines_list:
		try:
			p_name = p.split('/')[-1]
			if p_name in all_inputs:
				pipeline = importlib.reload(all_inputs_modules[p_name])
				if log:
					print("Reloading:", p_name)
			else:
				pipeline = importlib.import_module("pipelines." + p_name)
				if log:
					print("Loading:", p_name)
				
			p_exported = pipeline.exported_pipeline
			all_inputs[p_name] = p_exported
			all_inputs_modules[p_name] = pipeline
		except Exception as e:
			if log:
				print("FAILED: ", p, e)
				traceback.print_exc()
				
			pass
		finally:
			if log:
				print("-"*10)
	return all_inputs

if __name__=="__main__":
    print(get_all_inputs())