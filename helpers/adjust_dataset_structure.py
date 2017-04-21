import os
        # Script that automates the process of restructuring train dataset suitable for keras flow_from_directory API #

# Change the path according to the location of the folder...
path_to_train_folder = '/Users/shehabmohamed/Downloads/train'
nb_of_classes = 201 # + offset 1
list_of_wnids = [0] * nb_of_classes


def parse_wnids(filename='wnids_unsorted.txt'):
    print("Getting all class IDs...")
    with open(filename, mode='r') as input_file:
        for i, line in enumerate(input_file, 1):
            list_of_wnids[i] = line.strip('\n')
    print("Done parsing.")

def restructure_train_data():
    print("Restructuring train data...")
    for class_indx in range (1, nb_of_classes): # 200 classes + 1
        class_folder = '/' + str(list_of_wnids[class_indx])
        old_path_to_class_folder = path_to_train_folder + class_folder + '/images'
        new_path_to_class_folder = path_to_train_folder + class_folder
        os.system('cd ' + old_path_to_class_folder)
        for img in range(500):
            cmd = 'mv %s%s_%d.JPEG %s' % (old_path_to_class_folder, class_folder, img, new_path_to_class_folder)
            os.system(cmd)
        print("~~~~~~ %d - Moved all images from class %s" %(class_indx, class_folder))
        # delete empty images folder and boxes...
        os.system('rm -rf ' + old_path_to_class_folder)
        os.system('rm -rf ' + new_path_to_class_folder + class_folder + '_boxes.txt')
    print("Done Moving data from train/class/images to train/class.")

if __name__ == '__main__':
    # print("Current path: {}".format(os.getcwd()))
    parse_wnids()
    restructure_train_data()