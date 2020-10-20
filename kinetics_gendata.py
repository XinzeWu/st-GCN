
import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeder.feeder_kinetics import Feeder_kinetics

toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        num_person_in=1,  #observe the first 5 persons
        num_person_out=1,  #then choose 2 persons with the highest score
        max_frame=501):

    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)
    print("feeder: ", feeder)
    sample_name = feeder.sample_name
    print("sample_name: ", sample_name)  # 就是每个视频对应的json文件，例如'q8z5Zd5egBQ.json'
    sample_label = []

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 3, max_frame, 6, num_person_out))
    print("type(fp)", type(fp))
    for i, s in enumerate(sample_name):
        print("i, s:", i, s)
        data, label = feeder[i]
        print("data, label:", type(data), type(label))
        print("data.shape:", data.shape)
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(sample_name)))
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Eye-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='data/eye/eye-skeleton')
    parser.add_argument(
        '--out_folder', default='data/eye/eye-skeleton')
    arg = parser.parse_args()

    part = ['train', 'val']
    for p in part:
        data_path = '{}/eye_{}'.format(arg.data_path, p)
        label_path = '{}/eye_{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
    # data_path = 'F:/test_kinetics/kinetics_val'
    # label_path = 'F:/test_kinetics/kinetics_val_label.json'
    # data_out_path = 'F:/test_kinetics/test_result/val_data.npy'
    # label_out_path = 'F:/test_kinetics/test_result/val_label.pkl'
        gendata(data_path, label_path, data_out_path, label_out_path)