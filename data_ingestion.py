import argparse
from distutils.dir_util import copy_tree
from constants import *
from tqdm import tqdm
import cv2
import datetime
import pickle
import numpy as np
from keras.utils import np_utils
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str, required=True)
args = parser.parse_args()

PKL_PATH = os.path.join(DATASET_DIR, str(datetime.datetime.now()) + '.pkl')


dataset = {}

num_classes = 10

for ti, t in ((0, "train"), (1, "test")):
    folder_path = os.path.join(args.input_data, t)
    csv_file =  os.path.join(folder_path, "groundtruth.csv")
    images_len = len(glob.glob(os.path.join(folder_path, "*.png")))
    size = 32
    channels = 3
    dataset[t] = {
        'img': np.zeros(shape = [images_len, size, size, channels], dtype = float),
        'label': np.zeros(shape=[images_len],dtype = int),
        'class': []     # One hot encoded
    }
    
    f = open(csv_file, "r")
    for line_no, l in tqdm(enumerate(f.readlines()), total=images_len):
        img_file_path, label = l.split(",")
        #img_file_path = os.path.join(folder_path, img_file_path)
        img = cv2.imread(img_file_path)
        dataset[t]['img'][line_no] = img
        dataset[t]['label'][line_no] = int(label)

    dataset[t]['class'] = np_utils.to_categorical(dataset[t]['label'], num_classes)

print("Saving dataset to: {}".format(PKL_PATH))
with open(PKL_PATH, 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
