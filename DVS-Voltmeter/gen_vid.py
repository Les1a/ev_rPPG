import os

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def csv2sequence(csv_path):
    event_points = pd.read_csv(csv_path, names=['t', 'x', 'y', 'p'])  # names=['x', 'y', 'p', 't']

    fps = 1000
    period_t = 1 / fps  # s
    num_frame = ((event_points.iloc[-1]['t'] - event_points['t']) * fps) // 1e6

    '''sum events'''
    print(num_frame)
    img_list = []
    for n in np.arange(num_frame):
        print(n)
        chosen_idx = np.where((event_points['t'] >= period_t * n) * (event_points['t'] < period_t * (n + 1)))[0]
        xypt = event_points.iloc[chosen_idx]
        x, y, p = xypt['x'], xypt['y'], xypt['p']
        p = p * 2 - 1

        img = np.zeros((128, 128))
        img[y, x] += p
        img_list.append(img)
    img_list = np.array(img_list)
    return img_list


def txt2sequence(txt_path):
    event_points = pd.read_csv(txt_path, delim_whitespace=True, names=['t', 'x', 'y', 'p'], nrows=2.5e7)

    fps = 120
    period_t = 1 / fps * 1e6  # s
    num_frame = ((event_points.iloc[-1]['t'] - event_points['t'][0]) * fps) // 1e6

    '''sum events'''
    print(num_frame)
    img_list = []
    for n in np.arange(num_frame):
        # print(n)
        chosen_idx = np.where((event_points['t'] - event_points['t'][0] >= period_t * n) &
                              (event_points['t'] - event_points['t'][0] < period_t * (n + 1)))[0]
        xypt = event_points.iloc[chosen_idx]
        x, y, p = xypt['x'], xypt['y'], xypt['p']
        p = p * 2 - 1

        img = np.zeros((128, 128))
        img[y, x] += p
        img_list.append(img)
    img_list = np.array(img_list)
    return img_list


def vidvis(images, output_path='./vis.mp4'):
    # output_path = './vis.mp4'

    # define the color map
    cmap = plt.cm.get_cmap('jet').copy()
    cmap.set_under('k')

    # get the height, width, and number of frames of the video
    num_frames, height, width = images.shape

    # create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 60, (width, height))

    # write each image to the video writer
    for i in range(num_frames):
        # get the image and normalize it
        frame = images[i, :, :]
        image = cmap(frame)
        image = (image[:, :, :3] * 255).astype(np.uint8)

        # set the color for values below the minimum to black
        image[frame == 0] = [0, 0, 0]

        # set the color for values above or equal to 1 to red
        image[(frame >= 1)] = [0, 0, 255]

        # set the color for values below or equal to 1 to green
        image[(frame <= -1)] = [0, 255, 0]

        # write the color image to the video writer
        output_video.write(image)

    # release the video writer object
    output_video.release()


if __name__ == '__main__':
    # get sequence
    root_path = './'
    '''for dirname in os.listdir(root_path):
        if dirname.startswith('s') and not dirname.endswith('zip'):
            for filename in os.listdir(os.path.join(root_path, dirname)):
                if not filename.startswith('_') and filename.startswith('E'):
                    file_path = os.path.join(root_path, dirname, filename, filename + '.txt')
                    vid_path = file_path.replace('.txt', '.mp4')
                    img_seq = txt2sequence(file_path)

                    # trans to video
                    vidvis(img_seq, vid_path)

                    print(filename, ' Processing Done!')'''

    for filename in os.listdir(root_path):
        if not filename.startswith('_') and filename.endswith('1'):
            file_path = os.path.join(root_path, filename, filename + '.txt')
            vid_path = file_path.replace('.txt', '.mp4')
            img_seq = txt2sequence(file_path)

            # trans to video
            vidvis(img_seq, vid_path)

            print(filename, ' Processing Done!')
