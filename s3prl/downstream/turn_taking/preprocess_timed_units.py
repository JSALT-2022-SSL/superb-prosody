# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess_idx.py ]
#   Synopsis     [ preprocess xml file ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import xml.etree.ElementTree as ET
import csv

#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():

    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('-i', '--input_path', default='./Data/', type=str, help='Path to your maptask time unit', required=False)
    parser.add_argument('-o', '--output_path', default='./Data_out/', type=str, help='Path to store output', required=False)
    parser.add_argument('-a', '--time_extension', default='.xml', type=str, help='time unit file type (.xml)', required=False)
    parser.add_argument('-t', '--timestamp', default=50, type=int, help='timestamp you want to split', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for computation', required=False)

    args = parser.parse_args()
    return args

#########################
# GET METADATA AND SAVE #
#########################
def get_metadata_and_save(input_file, current_content, args):

    tree = ET.parse(input_file)
    root = tree.getroot()

    id_name = os.path.basename(input_file).split('.')[0]
    role_name = os.path.basename(input_file).split('.')[1]
    new_file_path = f"{args.output_path}/metadata/{id_name}.{role_name}.csv"
    os.makedirs((os.path.dirname(new_file_path)), exist_ok=True)

    f = open(new_file_path, 'w')
    writer = csv.writer(f)

    header = ['tag', 'start', 'end', 'utt', 'type', 'text']
    writer.writerow(header)

    for child in root:
        row = [child.tag, child.attrib.get('start'), child.attrib.get('end'),
               child.attrib.get('utt'), child.attrib.get('type'), child.text]
        writer.writerow(row)


##############################
# SPLIT TIMED UNITS AND SAVE #
##############################
def split_timed_units_and_save(input_file, current_content, args):

    tree = ET.parse(input_file)
    root = tree.getroot()

    id_name = os.path.basename(input_file).split('.')[0]
    role_name = os.path.basename(input_file).split('.')[1]
    new_file_path = f"{args.output_path}/{current_content}/{id_name}.{role_name}.csv"
    os.makedirs((os.path.dirname(new_file_path)), exist_ok=True)

    f = open(new_file_path, 'w')
    writer = csv.writer(f)

    header = ['start', 'end', 'target']
    writer.writerow(header)

    clock = 0
    for child in root: 
        start_time = float(child.attrib.get('start')) * 1000
        end_time = float(child.attrib.get('end')) * 1000
        target = 0
        if child.tag == 'tu':
            target = 1

        while (float(clock) < end_time):
            row = [clock, clock + args.timestamp, target]
            writer.writerow(row)
            clock += args.timestamp

    while clock % 60000 != 0:
        row = [clock, clock + args.timestamp, 0]
        writer.writerow(row)
        clock += args.timestamp


###################
# GET TIMED UNITS #
###################
def get_timed_units(args, tr_set, file_extension):

    for i, s in enumerate(tr_set):
        if os.path.isdir(os.path.join(args.input_path, s.lower())):
            s = s.lower()
        elif os.path.isdir(os.path.join(args.input_path, s.upper())):
            s = s.upper()
        else:
            assert NotImplementedError

        print('')
        todo = list(Path(os.path.join(args.input_path, s)).rglob('*' + file_extension)) # '*.flac'
        print(f'Preprocessing data in: {s}, {len(todo)} xml files found.')

        print('Splitting xml...', flush=True)
        Parallel(n_jobs=args.n_jobs)(delayed(get_metadata_and_save)(str(file), s, args) for file in tqdm(todo))
        Parallel(n_jobs=args.n_jobs)(delayed(split_timed_units_and_save)(str(file), s, args) for file in tqdm(todo))
    print('All done, saved at', args.output_path, 'exit.')


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()

    SETS = ['timed-units']

    for idx, s in enumerate(SETS):
        print('\t', idx, ':', s)
    tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
    tr_set = [SETS[int(t)] for t in tr_set.split(' ')]

    # Run split
    get_timed_units(args, tr_set, args.time_extension)


if __name__ == '__main__':
    main()
