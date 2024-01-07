import os
dirlist = []


def getFlist(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print('root_dir:', root)
        print('sub_dirs:', dirs)
        dirlist.append(dirs)
        print('files:', files)
    return files


resDir = './data_samples/interp/Cropped'
flist = getFlist(resDir)
print(dirlist)
