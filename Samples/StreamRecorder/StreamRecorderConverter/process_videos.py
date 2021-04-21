import tarfile
import os
import glob
from PIL import Image
import cv2
import imutils
import numpy as np
import argparse
from pathlib import Path

REFLECTIVITY = 'REFLECTIVITY'
PLY = 'PLY'
AVI = 'avi'

def process_videos(root_dir_path):
    print("PROCESS_VIDEOS: ", root_dir_path)

    depth_folder = None
    if os.path.exists(os.path.join(root_dir_path, 'Depth AHat')):
        depth_folder = 'Depth AHat'
    elif os.path.exists(os.path.join(root_dir_path, 'Depth Long Throw')):
        depth_folder = 'Depth Long Throw'

    if depth_folder is not None:
        if not os.path.isdir(os.path.join(root_dir_path,REFLECTIVITY)):
            os.mkdir(os.path.join(root_dir_path, REFLECTIVITY))
        glob_path = os.path.join(root_dir_path, depth_folder, '*_ab.pgm')
        files = glob.glob(glob_path)
        for i in range(len(files)):
            original_file_path = files[i]
            filename = original_file_path.split('\\')[-1]
            prefix = filename.split('.')[0]
            prefix = prefix[0:-3]
            new_path = os.path.join(root_dir_path, REFLECTIVITY, f'{prefix}.pgm')
            os.replace(original_file_path, new_path)

        if not os.path.isdir(os.path.join(root_dir_path, PLY)):
            os.mkdir(os.path.join(root_dir_path, PLY))
        glob_path = os.path.join(root_dir_path, depth_folder, '*.ply')
        files = glob.glob(glob_path)
        for i in range(len(files)):
            original_file_path = files[i]
            filename = original_file_path.split('\\')[-1]
            new_path = os.path.join(root_dir_path, PLY, filename)
            os.replace(original_file_path, new_path)



    #  folder, ext, fps, rotate, maxval)  
    # ('Depth AHaT','pgm', 60, None, 1055)

    sensors = [
        ('VLC RF','pgm',60, 'l', None, None),
        ('VLC LF','pgm',60, 'r', None, None),
        # ('VLC LR','pgm',30, 'r'),
        # ('VLC RR','pgm',30, 'r'),
        ('Depth Long Throw','pgm', 30, None, 7442, None),
        # ('long_throw_reflectivity','pgm',30, None),
        ('PV','png',60, None, None, None),
        ('Depth AHaT','pgm', 60, None, 1055, None),
        ('REFLECTIVITY','pgm',60, None, 10922, 'log')
    ]


    # initalize list to keep track of the min and max values for each sensor (to adjust the scaling)
    extremes = []
    for sensor_idx in range(len(sensors)):
        min_val = float('inf')
        max_val = -float('inf')
        extremes.append((min_val, max_val))



    for sensor_idx, sensor in enumerate(sensors):    
        name, filetype, fps, rotate, maxval, scale = sensor
        min_val, max_val = extremes[sensor_idx]

        print(sensor_idx, name)

        if not os.path.isdir(os.path.join(root_dir_path, AVI)):
            os.mkdir(os.path.join(root_dir_path, AVI))

        # if not os.path.isdir(f'./avi/{name}'):
        #     os.mkdir(f'./avi/{name}')
        
        
        # this block creates the video. Need to handle out of memory error

        img_array = []
        glob_path = os.path.join(root_dir_path, name, f'*.{filetype}')
        files = glob.glob(glob_path)
        sorted(files)
        first_timestamp = None
        
        previous_frame = None
        for i in range(0, len(files) - 1):
            fname = files[i]
            next_fname = files[i+1]
            
            timestamp = int(fname.split('\\')[-1].split('.')[0])       
            next_timestamp = int(next_fname.split('\\')[-1].split('.')[0])
            
            if first_timestamp is None:
                first_timestamp = timestamp

            # print('timestamp: ', timestamp)
            # print('next_timestamp', next_timestamp)

            # if timestamp < 132400215532731498:
            if i < 600:
                # print(i)
                
            # if True:
                img = cv2.imread(fname, -1)

                min_val = min(min_val, np.min(img))
                max_val = max(max_val, np.max(img))

                if maxval is not None:
                    raw = cv2.imread(fname, -1) / maxval

                    if scale == 'log':
                        raw = np.log(raw*1000 + 1) / np.log(1001)

                    gray = (raw*255).astype('uint8')
                    img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                
                else:
                    img = cv2.imread(fname)

                # print(img.shape)
                # print('min_val', min_val)
                # print('max_val', max_val)
                if rotate == 'r':
                    img = imutils.rotate_bound(img, 90)
                if rotate == 'l':
                    img = imutils.rotate_bound(img, -90)
                # cv2.imshow('a',img)
                # cv2.waitKey()
                if previous_frame is not None:
                    next_frame = round((timestamp - first_timestamp) / round(10000000 / fps))
                    for i in range(0, (next_frame - previous_frame) - 1):
                        img_array.append(img_array[-1])
                        # print("add")
                    # print("frame", next_frame)
                    img_array.append(img)
                    previous_frame = next_frame
                else:
                    img_array.append(img)
                    previous_frame = round((timestamp - first_timestamp) / round(10000000 / fps))
                # Image.open(f'./pgm/{name}/{f}').save(f'./png/{name}/image{i:05}.png')
                

        if len(img_array) > 0:
            height = img_array[0].shape[0]
            width = img_array[0].shape[1]

            size = (width,height)

            out = cv2.VideoWriter(os.path.join(root_dir_path, AVI, f'{name}.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()   

        # print(sensor_idx)
        extremes[sensor_idx] = (min_val, max_val)


    for i in range(len(sensors)):
        print(sensors[i][0], 'min: ', extremes[i][0], 'max: ', extremes[i][1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process recorded data.')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")

    args = parser.parse_args()

    root_dir_path = Path(args.recording_path)
    process_videos(root_dir_path)