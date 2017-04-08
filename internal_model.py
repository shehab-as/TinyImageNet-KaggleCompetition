import numpy as np
from keras.preprocessing import image as image_utils
# from keras.applications.
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
nb_of_classes = 201 # indexed by 1
class Model(object):
    def __init__(self):
        self.list_of_wnids = [0] * nb_of_classes
        self.Xtr = []
        self.Ytr = []
        self.predictions = []

    def parse_wnids(self, filename):
        with open(filename, mode='r') as input_file:
            for i, line in enumerate(input_file, 1):
                self.list_of_wnids[i] = line.strip('\n')
                print(" {} - {} ".format(i, line))

    def load_train_data(self):
        Xs_per_class = [0] * nb_of_classes
        Ys_per_class = [0] * nb_of_classes
        for i in range(1, nb_of_classes): # Looping Over 200 train image folders.
            xs = []
            ys = []
            folder_path = 'train/'+str(self.list_of_wnids[i])+'/images/'+str(self.list_of_wnids[i])+'_'
                # Loading all images of the same class into one chunk.
            for img_i in range(0, 500):
                img_path = folder_path + str(img_i) + '.JPEG'
                # Ex: train/n01443537/images/n01443537_0.JPEG
                print(" Loading {} ".format(img_path))

                img_x = image_utils.load_img(img_path, target_size=(64, 64))
                img_x = image_utils.img_to_array(img_x)
                # img_x = preprocess_input(img_x)
                # img_x = np.expand_dims(img_x, axis=0)
                #TODO: Preprocess image.
                img_y = str(self.list_of_wnids[i])
                xs.append(img_x)
                ys.append(img_y)

            # xs (500, 64, 64, 3)   tuple   PER CLASS
            # ys (500, )            tuple   PER CLASS
            Xs_per_class[i] = xs
            Ys_per_class[i] = ys

        # self.Xtr = np.reshape(Xs_per_class, (100000, 64, 64, 3)).astype("float")
        self.Xtr = Xs_per_class     # (200, 500, 64, 64, 3)
        self.Ytr = np.array(Ys_per_class)   # (200, 500, )
        print("Shape of Data -> {} - Shape of Label -> {}".format(np.shape(self.Xtr), np.shape(self.Ytr)))
        # Flatten the image to a vector [100000, 64*64*3]. UNCOMMENT BELOW.
        # self.Xtr = np.reshape(self.Xtr, (self.Xtr.shape[0], -1))


