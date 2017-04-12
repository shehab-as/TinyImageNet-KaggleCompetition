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
final_predictions = [0] * 10000

submission_number = 1
def create_submission_file():
    with open('my_submission%d.txt' %submission_number, 'w') as file:
        file.write('Id,Prediction\n')
        for i in range(10000):
            file.write('test_%d.JPEG,%s\n' % (i, final_predictions[i]))
        file.close()

#   Using Xception pre-trained Network...
pre_trained_model = Xception(weights='imagenet', input_tensor=input_tensor, include_top=False, pooling='avg')
x = pre_trained_model.output

#   freeze CNN layers...
for layer in pre_trained_model.layers:
    layer.trainable = False

#   Adding Fully Connected Layer on top of it...
predictions = Dense(nb_of_classes, activation='softmax')(x)

#   Actual Model
model = Model(inputs=[pre_trained_model.input], outputs=[predictions])      # Keras 2.0 API
adam = Adam(lr=0.0001)

#   Loading Weights...
print("Loading Best Accuracy Weights...")
# TODO: Load different weights and have different predictions...
model.load_weights("./")
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

# Classifying test_images
for i in range(10000):
    img_path = 'test_%d.JPEG' % i
    img = image.load_img(test_dir_path + img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    final_predictions[i] = preds.argmax()

# write to a submission file
create_submission_file()