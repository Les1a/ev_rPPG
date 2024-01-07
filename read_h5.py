import h5py
import cv2
import numpy as np
import os


def print_hdf5_structure(item, indent=0):
    """
    递归地打印HDF5文件的结构
    """
    if isinstance(item, h5py.File) or isinstance(item, h5py.Group):
        print(" " * indent + f"Group: {item.name}")
        for key in item.keys():
            print_hdf5_structure(item[key], indent + 2)
    elif isinstance(item, h5py.Dataset):
        print(" " * indent + f"Dataset: {item.name}")
    else:
        print(" " * indent + "Unknown type")


def read_hdf5_dataset(file, imgs_path):
    """
    读取HDF5文件中指定数据集的值
    """
    dataset_names = [name for name in file.keys() if isinstance(file[name], h5py.Dataset)]

    for dataset_name in dataset_names:
        dataset = file[dataset_name]
        data = dataset[:]

        if dataset_name == 'imgs':
            for i in range(len(data)):
                # data[i] = np.transpose(data[i], (1, 2, 0))
                cv2.imwrite(imgs_path+f'/image_{i:04d}.png', cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    root_path = './data/UBFC_h5'

    # path = './data/training/indoor_forward_3_davis_with_gt_0.h5'
    path = './data/UBFC_h5/1.h5'
    with h5py.File(path, 'r') as file:
        print_hdf5_structure(file)

    for file_path in os.listdir(root_path):
        imgs_path = './data/UBFC_img/' + file_path[:-3]
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)

        file_path = os.path.join(root_path, file_path)

        with h5py.File(file_path, 'r') as file:
            read_hdf5_dataset(file, imgs_path)

        print(imgs_path, ' Done!')
