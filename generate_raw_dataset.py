import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
import os
cifar10 = tf.keras.datasets.cifar10.load_data()

#print(len(cifar10[0][1]))
for ti, t in ((0, "train"), (1, "test")):
    folder_path = "data/raw_dataset/" + t + "/"
    os.makedirs(folder_path, exist_ok=True)
    csv_file =  folder_path + "groundtruth" + ".csv"
    with open(csv_file, "w") as f:
        for i in tqdm(range(len(cifar10[ti][0]))):
            file_path = folder_path + str(i).zfill(6) + ".png"
            f.write(file_path + "," + str(cifar10[ti][1][i][0]) + "\n")
            cv2.imwrite(file_path, cifar10[ti][0][i])