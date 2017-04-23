import numpy as np
from keras.preprocessing import image
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam, SGD
import time

now = time.strftime("%c")
nb_of_classes = 200
img_height = 299
img_width = 299
test_dir_path = 'test_images/'
input_tensor = Input(shape=(img_width, img_height, 3))
final_predictions = [0] * 10000
nb_of_test_images = 10000  # test/ 10,000

list_of_wnids = [0] * (nb_of_classes + 1)

def parse_wnids(filename='wnids.txt'):
    print("Getting all class IDs...")
    with open(filename, mode='r') as input_file:
        for i, line in enumerate(input_file, 1):
            list_of_wnids[i] = line.strip('\n')
    print("Done parsing.")

submission_number = 1
def create_submission_file():
    with open('my_submission%d.txt' %submission_number, 'w') as file:
        file.write('Id,Prediction\n')
        for i in range(nb_of_test_images):
            label_predict = list_of_wnids[final_predictions[i]]
            file.write('test_%d.JPEG,%s\n' % (i, label_predict))
        file.close()
    print("Done writing to file.")

#   Using Xception pre-trained Network...
pre_trained_model = Xception(weights='imagenet', input_tensor=input_tensor, include_top=False, pooling='avg')
x = pre_trained_model.output

predictions = Dense(nb_of_classes, activation='softmax')(x)

#   Actual Model
model = Model(inputs=[pre_trained_model.input], outputs=[predictions])      # Keras 2.0 API
adam = Adam(lr=0.0001)
sgd = SGD(lr=0.00001, momentum=0.8)

#   Loading Top Accuracy Weights and Compiling Model...
print("Loading Top Accuracy Weights...")
model.load_weights("./top_acc_weights.hdf5")
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

# Classifying test_images
for i in range(nb_of_test_images):
    img_path = 'test_%d.JPEG' % i
    img = image.load_img(test_dir_path + img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    final_predictions[i] = preds.argmax()

# write to a submission file
parse_wnids()
create_submission_file()