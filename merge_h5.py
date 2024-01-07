import os

import h5py
import csv
import numpy as np


# 从CSV文件读取文本数据
def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        events_array = np.array([list(map(int, row)) for row in reader], dtype=int)

    return events_array


# 将CSV数据转换为H5格式
def merge_ev_h5(data, h5_file):
    with h5py.File(h5_file, 'a') as hf:
        events_group = hf.create_group('/events')
        ps_dataset = events_group.create_dataset('ps', data=data[:, 3])
        ts_dataset = events_group.create_dataset('ts', data=data[:, 0])
        xs_dataset = events_group.create_dataset('xs', data=data[:, 1])
        ys_dataset = events_group.create_dataset('ys', data=data[:, 2])


def recursive_delete(group):
    for key in list(group.keys()):
        if isinstance(group[key], h5py.Group):
            recursive_delete(group[key])
        elif isinstance(group[key], h5py.Dataset):
            del group[key]


def del_h5(file_path):
    with h5py.File(file_path, 'a') as hf:
        if '/imgs' in hf:
            group = hf['/imgs']
            recursive_delete(group)
            del hf['/imgs']
            print(file_path, "Dataset '/imgs' deleted.")


if __name__ == '__main__':
    # 指定您的CSV文件路径
    ev_file_path = './data/UBFC_ev'
    h5_file_path = './data/UBFC_h5'

    for file in os.listdir(ev_file_path):
        ev_file = os.path.join(ev_file_path, file)
        h5_file = os.path.join(h5_file_path, file.replace('txt', 'h5'))

        # 读取CSV文件并保存为H5
        ev_data = read_csv_file(ev_file)
        merge_ev_h5(ev_data, h5_file)

        print(file, ' done!')

    '''for file in os.listdir(ev_file_path):
        h5_file = os.path.join(h5_file_path, file.replace('txt', 'h5'))
        del_h5(h5_file)'''

