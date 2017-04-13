nb_of_classes = 201 # + offset 1
list_of_wnids = [0] * nb_of_classes

def parse_wnids(filename='wnids.txt'):
    print("Getting all class IDs...")
    with open(filename, mode='r') as input_file:
        for i, line in enumerate(input_file, 1):
            list_of_wnids[i] = str(line.strip('\n'))
            # print(" {} - {} ".format(i, list_of_wnids[i]))
    del list_of_wnids[0]    # Removing empty first element.
    print("Done parsing.")

def sort_wnids():
    list_of_wnids.sort()
    with open('wnids_sorted.txt', mode='w') as write_file:
        for i, line in enumerate(list_of_wnids, 1):
            write_file.write(str(line) + '\n')
    print("Done writing.")

if __name__ == '__main__':
    parse_wnids()
    sort_wnids()