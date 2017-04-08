import numpy as np
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import time

now = time.strftime("%c")
nb_of_classes = 200

    ###     Using Xception pre-trained Network      ###
pre_trained_model = Xception(weights='imagenet', include_top=False)
x = pre_trained_model.output
    ###     Adding Layer on top of it      ###
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_of_classes, activation='softmax')(x)

    ###     Actual Model   ###
model = Model(input=pre_trained_model.input, output=predictions)
adam = Adam(lr=0.0001)

                            # Loading Weights...
print("Loading Best Accuracy Weights...")
model.load_weights("./.hdf5")
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])


# n_labels = nb_of_classes
#
# X_test = X_test - np.mean(X_test, axis=0)
# X_test /= np.std(X_test, axis=0)
#
# n_imgs_by_label = np.zeros(n_labels, dtype=np.dtype(int))
# n_top1_accurate_by_label = np.zeros(n_labels, dtype=np.dtype(int))
#
# for i, img in enumerate(X_test):
#     ground_truth = Y_test[i].argmax()
#     n_imgs_by_label[ground_truth] += 1
#
#     img = np.expand_dims(img, axis=0)
#     preds = model.predict(img)
#
#     if ground_truth == preds.argmax():
#         n_top1_accurate_by_label[ground_truth] += 1
#
# for i in range(len(n_top1_accurate_by_label)):
#     accuracy_of_class_i = (n_top1_accurate_by_label[i]/1000)
#     print("Accuracy of Class[{}] => {}".format(i, accuracy_of_class_i))
#
# # scores = model.evaluate_generator(test_datagen, val_samples=10000)
# #
# # print(model.metrics_names, scores)
# # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))