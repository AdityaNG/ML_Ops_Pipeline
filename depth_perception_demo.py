import pandas as pd
import os
from pipeline_input import *
from constants import *

class depth_interp_airsim(pipeline_dataset_interpreter):
	def load(self) -> None:
		super().load()
		airsim_txt=os.path.join(self.input_dir, 'airsim_rec.txt')
		images_folder=os.path.join(self.input_dir, 'images')
		test_path=os.path.join(self.input_dir, 'Test/Test/JPEGImages')
		test_annot=os.path.join(self.input_dir, 'Test/Test/Annotations')
		assert os.path.exists(airsim_txt)
		assert os.path.exists(images_folder)
		raw_data = pd.read_csv(airsim_txt, sep='\t')

		self.dataset = {
			'train': {
				'x': xtrain["image"].unique(),
				'y': xtrain
			},
			'test': {
				'x': xtest["image"].unique(),
				'y': xtest
			}
		}

class depth_data_visualizer(pipeline_data_visualizer):

	def visualize(self, x, y, preds, mode='') -> None:
		pass

class depth_evaluator:

	def evaluate(self, x, y, plot=False):
		preds = self.predict(x)
		results = 0
		# TODO: implement evaluation
		return results, preds

class depth_pipeline_model(depth_evaluator, pipeline_model):

	def load(self):
		self.model = 0
		
	def train(self, dataset):
		# TODO: Training
		pass
		
	def predict(self, x: dict) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		predict_results = {
			'xmin': [], 'ymin':[], 'xmax':[], 'ymax':[], 'confidence': [], 'name':[], 'image':[]
		}
		# TODO: Implement prediction
		predict_results = pd.DataFrame(predict_results)
		return predict_results


depth_input = pipeline_input("depth_det", {'depth_interp_airsim': depth_interp_airsim}, 
	{
		'depth_pipeline_model': depth_pipeline_model,
	}, dict(), dict())