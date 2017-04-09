import os

        # Script that automates the process of creating a validation dataset from train data #
        # Hardcoded to create 20% of train data --> 20,000 images total in validation folder #

path_to_validation_folder = '/Users/shehabmohamed/PycharmProjects/TinyImageNet-KaggleCompetition/data/validation'
path_to_train_folder = '/Users/shehabmohamed/PycharmProjects/TinyImageNet-KaggleCompetition/data/train'
nb_of_classes = 200
list_of_wnids = [0] * (nb_of_classes + 1)

def parse_wnids(filename='wnids.txt'):
    print("Getting all class IDs...")
    with open(filename, mode='r') as input_file:
        for i, line in enumerate(input_file, 1):
            list_of_wnids[i] = line.strip('\n')
            # print(" {} - {} ".format(i, list_of_wnids[i]))

def create_validation_folder(foldername='validation'):
    os.system('mkdir ' + path_to_validation_folder)
    for class_indx in range (1, nb_of_classes):
        for img in range (100):
            os.chdir(path_to_validation_folder)
            cmd = 'mkdir ' + str(list_of_wnids[class_indx])
            os.system(cmd)


if __name__ == '__main__':
    print("Current path: {}".format(os.getcwd()))
    # parse_wnids()
    # create_validation_folder()