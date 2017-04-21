import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3, Xception
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import time

now = time.strftime("%c")
nb_of_classes = 200
img_height = 299
img_width = 299
batch_size = 32
nb_of_epochs = 100
nb_train_samples = 90000
nb_validation_samples = 10000
train_dir_path = 'data/train'
validate_dir_path = 'data/validation'


class CustomImageDataGen(ImageDataGenerator):  # Overloading the ImageDataGenerator
    def standardize(self, x):
        if self.featurewise_center:
            x /= 255.
            x -= 0.5
            x *= 2.
        return x

# Fixed Seed and callbacks...
np.random.seed(seed=204)
tensorboard_callback = TensorBoard(log_dir="./logs/training_" + now, histogram_freq=0, write_graph=True,
                                   write_images=False)
checkpoint_callback = ModelCheckpoint(filepath="./top_acc_weights.hdf5", verbose=1, save_best_only=True,
                                      monitor="val_acc")
reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, verbose=0, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0.000001)

callbacks = [tensorboard_callback, checkpoint_callback, reduce_on_plateau]
input_tensor = Input(shape=(img_width, img_height, 3))

#   Using Xception pre-trained Network...
# pre_trained_model = InceptionV3(weights='imagenet', input_tensor=input_tensor, include_top=False, pooling='avg')
pre_trained_model = Xception(weights='imagenet', input_tensor=input_tensor, include_top=False, pooling='avg')
x = pre_trained_model.output

#   freeze CNN layers...
for layer in pre_trained_model.layers[:20]:
    layer.trainable = False

predictions = Dense(nb_of_classes, activation='softmax')(x)

#   Building Model...
model = Model(inputs=[pre_trained_model.input], outputs=[predictions])  # Keras 2.0 API
# model = Model(input=pre_trained_model.input, output=predictions)          # Keras 1.0 API

adam = Adam(lr=0.0001)  # not really good optimizer...
# Phase 1 for SGD (High Learning Rate) lr= 0.01
# Phase 2 for SGD (Slow Learning Rate) lr= 0.00001
sgd = SGD(lr=0.00001, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#   Loading Data...
#   Data Generation...
train_data_generator = CustomImageDataGen(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # don't set each sample mean to 0
    featurewise_std_normalization=True,  # divide all inputs by std of the dataset
    samplewise_std_normalization=False,  # don't divide each input by its std
    zca_whitening=False,  # don't apply ZCA whitening.
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180).
    horizontal_flip=True,  # randomly flip horizontal images.
    vertical_flip=False,  # don't randomly flip vertical images.
    zoom_range=0.1,  # slightly zoom in.
    width_shift_range=0.1,
    height_shift_range=0.1
)
validate_data_generator = CustomImageDataGen(
    featurewise_center=True,  # test images should have input mean set to 0 over the images.
    featurewise_std_normalization=True,  # test images should have all divided by std of the images.
    zca_whitening=False
)

#   Loading Train Data...
train_data_generator = train_data_generator.flow_from_directory(train_dir_path, target_size=(img_width, img_height),
                                                                batch_size=batch_size,
                                                                shuffle=True)
#   Loading Validation Data...
validate_data_generator = validate_data_generator.flow_from_directory(validate_dir_path,
                                                                      target_size=(img_width, img_height),
                                                                      batch_size=batch_size,
                                                                      shuffle=False)
print("Dataset loaded!")
print("Now Training...")
model.fit_generator(train_data_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=nb_of_epochs,
                    validation_data=validate_data_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    verbose=1, callbacks=callbacks)
