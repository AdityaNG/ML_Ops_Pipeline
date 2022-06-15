import pandas as pd
import os
import PIL
import cv2
from pipeline_input import *
from constants import *

class depth_interp_airsim(pipeline_dataset_interpreter):
	NUM_CAMS = 0
	mode_name = {
        0: 'Scene', 
        1: 'DepthPlanar', 
        2: 'DepthPerspective',
        3: 'DepthVis', 
        4: 'DisparityNormalized',
        5: 'Segmentation',
        6: 'SurfaceNormals',
        7: 'Infrared'
    }
	cam_name = {
        '0': 'FrontL'
    }

	def load(self) -> None:
		super().load()
		airsim_txt=os.path.join(self.input_dir, 'airsim_rec.txt')
		images_folder=os.path.join(self.input_dir, 'images')
		
		assert os.path.exists(airsim_txt)
		assert os.path.exists(images_folder)

		df = pd.read_csv(airsim_txt, sep='\t')
		df.set_index('TimeStamp')

		self.NUM_CAMS = len(df["ImageFile"].iloc[0].split(";"))

		self.cam_name = {
			'0': 'FrontL',
			str(self.NUM_CAMS): 'FrontR'
		}
		for i in range(1, self.NUM_CAMS):
			self.cam_name[str(i)] = 'C' + str(i)

		x_vals = dict()
		for col in df.columns:
			if col!="ImageFile":
				x_vals[col] = []
		x_vals["ImageFile"] = []

		for index, row in df.iterrows():
			files = row["ImageFile"].split(";")
			input_images = []
			gt_images = []
			for f in files:
				cam_id, img_format = f.split("_")[2:4]
				# if img_format in (1,2,3,4,5,6):
				# 	gt_images.append(os.path.join(images_folder, f))
				# 	assert os.path.exists(gt_images[-1])
				if img_format=='0':
					input_images.append(os.path.join(images_folder, f))
					assert os.path.exists(input_images[-1])
			for col in df.columns:
				if col!="ImageFile":
					x_vals[col] += [row[col]]
			x_vals["ImageFile"] += [";".join(input_images)]
		x_vals = pd.DataFrame(x_vals)
		self.dataset = {
			'train': {
				'x': x_vals,
				'y': df
			},
			'test': {
				'x': x_vals,
				'y': df
			}
		}


class depth_data_visualizer(pipeline_data_visualizer):

	def visualize(self, x, y, preds, mode='') -> None:
		pass

class depth_evaluator:

	def evaluate(self, x: pd.DataFrame, y, plot=False):
		preds = 0
		for inxex, row in x.iterrows():
			pred = self.predict(row)
		
		results = 0
		# TODO: implement evaluation
		return results, preds

class depth_pipeline_model(depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2()
		
	def train(self, dataset):
		# TODO: Training
		pass
		
	def predict(self, x) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		predict_results = {
			'xmin': [], 'ymin':[], 'xmax':[], 'ymax':[], 'confidence': [], 'name':[], 'image':[]
		}
		# TODO: Implement prediction
		print(x)
		image_files = x["ImageFile"].split(";")
		for image_path in image_files:
			print(image_path)
			f = image_path.split("/")[-1]
			cam_id, img_format = f.split("_")[2:4]
			img_format = int(img_format)
			if f.endswith('.ppm'):
				#img = PIL.Image.open(image_path)
				#img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
				img = cv2.imread(image_path)
			#elif f.endswith('.pfm'):
			#	img, scale = airsim.read_pfm(f)
			else:
				print("Unknown format")

			depth = self.model.eval(img)
			cv2.imshow('depth_'+str(cam_id),depth)
		cv2.waitKey(1)
		predict_results = pd.DataFrame(predict_results)
		return predict_results


depth_input = pipeline_input("depth_det", {'depth_interp_airsim': depth_interp_airsim}, 
	{
		'depth_pipeline_model': depth_pipeline_model,
	}, dict(), dict())