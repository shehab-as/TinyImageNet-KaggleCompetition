import numpy as np
import _pickle as Pickle
from PIL import Image
filename = 'wnids.txt'


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
        for i in range(1, 201):
            Xs = []
            Ys = []
            folder_path = '/train/'+str(self.list_of_wnids[i])+'/images/'+str(self.list_of_wnids[i])+'_'
            for img_i in range(0, 500):
                img = folder_path + str(img_i) + '.JPEG'
                print(" {} ".format(img))
                # Ex: train/n01443537/images/n01443537_0.JPEG
                with open(img, mode='rb') as imgfile:
                    img_x = Image.open(imgfile)
                    # img_x = Pickle.load(img, encoding='bytes')
                    img_y = str(self.list_of_wnids[i])
                    Xs.append(img_x)
                    Ys.append(img_y)
                    # img_x = img_x.reshape(500, 3, 64, 64).transpose(0, 2, 3, 1).astype("float")

# /Users/shehabmohamed/PycharmProjects/TinyImageNet-KaggleCompetition/train/n02124075/images/n02124075_0.JPEG
m = Model()
m.parse_wnids(filename)
# m.load_train_data()
