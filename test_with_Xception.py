import numpy as np
from keras.preprocessing import image
from keras.applications import Xception
from keras.applications.xception import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
import time

now = time.strftime("%c")
nb_of_classes = 200
img_height = 299
img_width = 299
test_dir_path = 'test_images/'  # test path not used yet...
input_tensor = Input(shape=(img_width, img_height, 3))

#   Using Xception pre-trained Network...
pre_trained_model = Xception(weights='imagenet', input_tensor=input_tensor, include_top=False)
x = pre_trained_model.output
x = GlobalAveragePooling2D()(x)

#   freeze CNN layers...
for layer in pre_trained_model.layers:
    layer.trainable = False

#   Adding Fully Connected Layer on top of it...
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_of_classes, activation='softmax')(x)

#   Actual Model
model = Model(inputs=[pre_trained_model.input], outputs=[predictions])      # Keras 2.0 API
adam = Adam(lr=0.0001)

#   Loading Weights...
print("Loading Best Accuracy Weights...")
model.load_weights("./top_acc_weights.hdf5.hdf5")
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

final_predictions = [0] * 10000
for i in range(10000):
    img_path = 'test_%d.JPEG' % i
    img = image.load_img(test_dir_path + img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    final_predictions[i] = preds.argmax()

    # print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
