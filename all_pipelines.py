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

	#pipelines_list = os.listdir('pipelines')
	pipelines_list = list(map(lambda x: x.split("/")[-1],glob.glob('pipelines/*.py')))
	if log:
		print(pipelines_list)
		print("-"*10)

	for p in pipelines_list:
		if p!="template.py":
			try:
				p_name = p.split(".py")[-2]
				if p_name in all_inputs:
					pipeline = importlib.reload(all_inputs_modules[p_name])
					if log:
						print("Reloading:", p_name)
				else:
					pipeline = importlib.import_module("pipelines." + p_name)
					if log:
						print("Loading:", p_name)
				
				p_exported = pipeline.exported_pipeline
				print("pipeline_hash(",p_name,")=",hash(p_exported))
				#all_inputs[p_exported.get_pipeline_name()] = p_exported
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