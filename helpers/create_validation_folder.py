import os
        # Script that automates the process of creating a validation dataset from train data #
        # Hardcoded to create 20% of train data --> 20,000 images total in validation folder #

path_to_validation_folder = '/Users/shehabmohamed/PycharmProjects/TinyImageNet-KaggleCompetition/data/validation'
path_to_train_folder = '/Users/shehabmohamed/PycharmProjects/TinyImageNet-KaggleCompetition/data/train'
nb_of_classes = 201 # + offset 1
list_of_wnids = [0] * nb_of_classes
percentage = 0.2
validation_size = (percentage * 100000) / (nb_of_classes - 1)

def parse_wnids(filename='wnids.txt'):
    print("Getting all class IDs...")
    with open(filename, mode='r') as input_file:
        for i, line in enumerate(input_file, 1):
            list_of_wnids[i] = line.strip('\n')
            # print(" {} - {} ".format(i, list_of_wnids[i]))
    print("Done parsing.")

def create_validation_folder():
    print("Creating Validation folder...")
    os.system('mkdir ' + path_to_validation_folder)
    for class_indx in range (1, nb_of_classes): # 200 classes + 1
        class_folder = '/' + str(list_of_wnids[class_indx])
        print("%d - making val folder for class %s" %(class_indx, class_folder))
        cmd = 'mkdir ' + path_to_validation_folder + class_folder
        os.system(cmd)
        for img in range (validation_size):     # 100/500 imgs per class (20%)
            # /Users/shehabmohamed/PycharmProjects/TinyImageNet-KaggleCompetition/data/train/n01443537/n01443537_0.JPEG
            cmd = 'mv %s%s%s_%d.JPEG %s%s' % (path_to_train_folder, class_folder, class_folder, img,
                                              path_to_validation_folder, class_folder)
            os.system(cmd)
    print("Done creating validation data.")


if __name__ == '__main__':
    print("Current path: {}".format(os.getcwd()))
    parse_wnids()
    create_validation_folder()