##80:20 training-testing split

import os
import glob
import csv
import argparse
from sklearn.model_selection import train_test_split

#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
    
    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('-i', '--input_path', default='./Data/', type=str, help='Path to your maptask time unit', required=False)
    parser.add_argument('-o', '--output_path', default='./Data_out/', type=str, help='Path to store output', required=False)
    
    args = parser.parse_args()
    return args


###################
# GET TIMED UNITS #
###################
def get_split_table(args, element):

    full_path_names = glob.glob(os.path.join(f"{args.input_path}/{element}", "*"))
    id_names = [os.path.basename(path_name).split('.')[0] for path_name in full_path_names]
    id_names.remove('q6ec2') # data with missing values
    raw_train_ids, test_ids = train_test_split(id_names, train_size=0.8, random_state=0)
    train_ids, dev_ids = train_test_split(raw_train_ids, train_size=0.9, random_state=0)

    train_ids_file = f"{args.output_path}/split-tables/train_ids.csv"
    os.makedirs((os.path.dirname(train_ids_file)), exist_ok=True)
    f = open(train_ids_file, 'w')
    writer = csv.writer(f)
    header = ['ids']
    writer.writerow(header)
    for train_id in train_ids:
        row = [train_id]
        writer.writerow(row)
    
    dev_ids_file = f"{args.output_path}/split-tables/dev_ids.csv"
    os.makedirs((os.path.dirname(dev_ids_file)), exist_ok=True)
    f = open(dev_ids_file, 'w')
    writer = csv.writer(f)
    header = ['ids']
    writer.writerow(header)
    for dev_id in dev_ids:
        row = [dev_id]
        writer.writerow(row)

    test_ids_file = f"{args.output_path}/split-tables/test_ids.csv"
    os.makedirs((os.path.dirname(test_ids_file)), exist_ok=True)
    f = open(test_ids_file, 'w')
    writer = csv.writer(f)
    header = ['ids']
    writer.writerow(header)
    for test_id in test_ids:
        row = [test_id]
        writer.writerow(row)

        
########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()

    SETS = ['dialogues']
    
    for element in SETS:
        get_split_table(args, element)


if __name__ == '__main__':
    main()

