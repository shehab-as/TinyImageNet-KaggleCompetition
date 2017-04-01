import numpy as np
from keras.preprocessing import image as image_utils

import keras
import tensorflow
import _pickle as Pickle
from PIL import Image

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# import os
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in '%s': %s" % (cwd, files))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# TODO: Start with Loading VGG16 Network.
# print("[INFO] loading network...")
# model = VGG16(weights="imagenet")

class Model(object):
    def __init__(self):
        pass

    def parse_wnids(self, filename):
        self.list_of_wnids = [0]*201
        with open(filename, mode='r') as input_file:
            for i, line in enumerate(input_file, 1):
                self.list_of_wnids[i] = line.strip('\n')
                print(" {} - {} ".format(i, line))

    def load_train_data(self):
        self.X = []
        self.Y = []
        for i in range(1, 201): # Looping Over 200 train image folders.
            Xs = []
            Ys = []
            folder_path = 'train/'+str(self.list_of_wnids[i])+'/images/'+str(self.list_of_wnids[i])+'_'
                # Loading all images of the same class into one chunk.
            for img_i in range(0, 500):
                img_path = folder_path + str(img_i) + '.JPEG'
                # Ex: train/n01443537/images/n01443537_0.JPEG
                print(" Loading {} ".format(img_path))

                img_x = image_utils.load_img(img_path, target_size=(64, 64))
                img_x = image_utils.img_to_array(img_x)
                img_x = np.expand_dims(img_x, axis=0)
                #TODO: Preprocess image.

                img_y = str(self.list_of_wnids[i])
                Xs.append(img_x)
                Ys.append(img_y)

        # self.X =
        # self.Y = np.array(Ys)

