import os
import argparse


def pre_process(folder_path):
    i = 0
    file_content = []
    file_list = os.listdir(folder_path)
    flie_list = file_list.sort()
    for filename in file_list:
        if filename.endswith('png') or filename.endswith('jpg'):
            file_path = os.path.join(folder_path, filename)

            timestamp = str(int((int(filename[-8:-4])) * 1e6 * 1 / 30 + 1000)).zfill(12)

            file_content.append(f"{file_path} {timestamp}\n")
            i += 1

    output_file_path = folder_path + '/info.txt'
    with open(output_file_path, 'w') as file:
        file.writelines(file_content)

    print(f"File '{output_file_path}' has been created")


if __name__ == '__main__':

    root_path = './data_samples/interp'

    '''for i in range(4, 27):
        root_path = './data_samples/interp/Cropped/sub' + str(i).zfill(2)'''
    '''for folder_name in os.listdir(root_path):
        if not folder_name.startswith('_'):
            pre_process(os.path.join(root_path, folder_name))
'''

    parser = argparse.ArgumentParser(description='data path for synthetic.')
    parser.add_argument('--dir_path', '-path', type=str, default="../data/UBFC_img", help='folder path.')

    args = parser.parse_args()
    dir_path = args.dir_path

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        pre_process(file_path)

    
