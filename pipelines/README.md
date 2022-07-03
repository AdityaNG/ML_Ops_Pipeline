# Creating a custom pipeline

To create a custom pipeline, one must first implement the following classes:
```python
class template_interp_1(pipeline_dataset_interpreter):
	pass
class template_data_visualizer(pipeline_data_visualizer):
	pass
class template_evaluator:
	pass
class template_pipeline_model(template_evaluator, pipeline_model):
	pass
class template_pipeline_ensembler_1(template_evaluator, pipeline_ensembler):
	pass

```

They must then create a pipeline object and specify the set of each class. Note that each class can have multiple implementations. One pipeline may have multiple datasets, models, ensembles, evaluation and visualisation scripts. Finally, the pipeline object must be exported as `exported_pipeline` and the file must be placed within the `pipeline/` directory so it may be automatically picked up by the `all_pipelines.py` script which scans for pipelines

```python
template_input = pipeline_input("template", 
	{
		'karthika95-pedestrian-detection': template_interp_1
	}, {
		'template_pipeline_model': template_pipeline_model,
	}, {
		'template_pipeline_ensembler_1': template_pipeline_ensembler_1
	}, {
		'template_data_visualizer': template_data_visualizer
	})

# Write the pipeline object to exported_pipeline
exported_pipeline = template_input
```

To validate if you pipeline has been exported correctly, run the following. Any errors faced while processing your pipeline will be output here. For example here, the `cifar10_cnn.py` has thrown an error and the stacktrace has been printed
```console
lxd1kor@BANI-C-0069L:~/ML_Ops_Pipeline$ python all_pipelines.py 
['obj_det.py', 'template.py', 'depth_det.py', 'cifar10_cnn.py']
-----------------------------------------------------------------------------
Loading:  obj_det.py
2022-06-27 13:59:59.291422: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
-----------------------------------------------------------------------------
Loading:  depth_det.py
-----------------------------------------------------------------------------
Loading:  cifar10_cnn.py
FAILED:  cifar10_cnn.py __init__() missing 1 required positional argument: 'p_vizualizer'
Traceback (most recent call last):
  File "all_pipelines.py", line 23, in <module>
    pipeline = importlib.import_module("pipelines." + p_name)
  ...
TypeError: __init__() missing 1 required positional argument: 'p_vizualizer'
-----------------------------------------------------------------------------
{'obj_det': <pipeline_input.pipeline_input object at 0x7f0552c66d60>, 'depth_det': <pipeline_input.pipeline_input object at 0x7f03cebada90>}
```

# Dataset Interpreter

Multiple datasets can be given as input to the pipeline. But all of them may not have the same interface. Hence it is important to unify them into one common dataset structure. Consider the task of object detection, lets say we start using the following datasets:
1. KITTI 2015
2. KITTI 2012
3. Coco Dataset

All three of the above datasets have slightly different formats they store the ground truths in. But we know that the common inputs from the datasets would be RGB images and the common outputs from the dataset would be a set of bounding box associated with each image. We implement the following interfaces to unify the 3 datasets:

```python
class KITTI2015_interp(pipeline_dataset_interpreter):
	def load(self) -> None:
		print("Loading KITTI2015 from:", self.input_dir)
		# TODO
		self.dataset = {
			'train': {'x': [],'y': []},
			'test': {'x': [],'y': []}
		}

class KITTI2012_interp(pipeline_dataset_interpreter):
	def load(self) -> None:
		print("Loading KITTI2012 from:", self.input_dir)
		# TODO
		self.dataset = {
			'train': {'x': [],'y': []},
			'test': {'x': [],'y': []}
		}

class COCO_interp(pipeline_dataset_interpreter):
	def load(self) -> None:
		print("Loading COCO from:", self.input_dir)
		# TODO
		self.dataset = {
			'train': {'x': [],'y': []},
			'test': {'x': [],'y': []}
		}
```

The other classes (model, ensemble, visualizer) will be using multiple datasets without the knowledge that they are because they will be given a common interface to interact with all the datasets. None of the other classes must be worried about the dataset structure, all the File IO must be handled by the dataset interpreter and all that must be visible to the other classes that use the

## Dataset Interpreter Interface Template

```python
class template_interp_1(pipeline_dataset_interpreter):
	def load(self) -> None:
		super().load()
		print("Loading from:", self.input_dir)
		# TODO
		self.dataset = {
			'train': {'x': [],'y': []},
			'test': {'x': [],'y': []}
		}
```

# Visualizer Interface

A pipeline may have a set of visualizers. For the case of object detection, the following visualisers may be made:
1. Failure cases (IOU=0.0)
2. Poor Performers (IOU<=0.5)
3. Input to Model Prediction mapping
4. Model vs Dataset perfornances
5. Model Training+Validation loss graph

## Visualizer Interface Template

```python
class template_data_visualizer(pipeline_data_visualizer):

	def visualize(self, x, y, preds, save_dir) -> None:
		# TODO: Visualize the data
		print(x)
```

# Model and Ensemble Interfaces

A model must implement the following functions:
1. load: Loads the architecture and model weights if any
2. predict: output of the model given a list of inputs
3. evaluate: Produces evaluation metrics for a given set of inputs
4. train: train the model on the given dataset, save weights (TODO)

Note that it helps to have a common evaluation script for all the models and ensembles

## Model and Ensemble Interface Template

```python
class template_model(pipeline_model):
	def load(self) -> None:
		# Load the model into self.model
		pass

	def predict(self, x: np.array):
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		pass

	def evaluate(self, x, y):
		pass

	def train(self, x, y):
		pass
```

Template with common evluation script:
```python
class template_evaluator:

	def evaluate(self, x, y):
		preds = self.predict(x)
		# TODO: write a common evaluation script that is common to all models
		# Note: Optionally, you can give model specific implementations for the evaluation logic
		#		by overloading the evaluate(self, x, y) method in the model class
		results = {
			'some_metric': 0,
			'another_metric': 0
		}
		return results, preds


class template_pipeline_model(template_evaluator, pipeline_model):

	def load(self):
		# TODO: Load the model
		self.model = None
		
	def train(self, x, y) -> np.array:
		preds = self.predict(x)
		# TODO: Train the model		
		results = {
			'training_results': 0,
		}
		return results, preds

	def predict(self, x: dict) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		predict_results = {
			'result': [], 'another_result': []
		}
		for i in tqdm(x):
			# TODO produce model predictions
			predict_results["result"] += ["Some Result"]
			predict_results["another_result"] += ["Another Result"]
				
		predict_results = pd.DataFrame(predict_results)
		return predict_results


class template_pipeline_ensembler_1(template_evaluator, pipeline_ensembler):

	def predict(self, x: dict) -> np.array:
		model_names = list(x.keys())
		predict_results = {
			'result': [], 'another_result': []
		}
		for i in tqdm(x):
			for mod_name in model_names:
				preds = x[mod_name]
			# TODO produce ensebled results based on all the model's predictions
			predict_results["result"] += ["Some Ensembled Result"]
			predict_results["another_result"] += ["Another Ensembled Result"]
				
		predict_results = pd.DataFrame(predict_results)
		return predict_results


	def train(self, x, y) -> np.array:
		preds = self.predict(x)
		# TODO: train ensemble model
		results = {
			'training_results': 0,
		}
		return results, preds
```
