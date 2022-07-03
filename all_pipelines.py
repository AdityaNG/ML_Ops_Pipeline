import glob
import os
import importlib
import traceback

log = False
if __name__=="__main__":
    log = True

def get_all_inputs():
	all_inputs = {}

	#pipelines_list = os.listdir('pipelines')
	pipelines_list = list(map(lambda x: x.split("/")[-1],glob.glob('pipelines/*.py')))
	if log:
		print(pipelines_list)
		print("-"*10)

	for p in pipelines_list:
		if p!="template.py":
			if log:
				print("Loading: ", p)
			try:
				p_name = p.split(".py")[-2]
				pipeline = importlib.import_module("pipelines." + p_name)
				p_exported = pipeline.exported_pipeline
				all_inputs[p_exported.get_pipeline_name()] = p_exported
			except Exception as e:
				if log:
					print("FAILED: ", p, e)
					traceback.print_exc()
					
				pass
			finally:
				if log:
					print("-"*10)
	return all_inputs
