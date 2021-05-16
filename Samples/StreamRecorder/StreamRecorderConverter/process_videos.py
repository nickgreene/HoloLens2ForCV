import tarfile
import os
import glob
from PIL import Image
import cv2
import imutils
import numpy as np
import argparse
from pathlib import Path
import ffmpeg

REFLECTIVITY = 'REFLECTIVITY'
PLY = 'PLY'
VIDEOS = 'videos'
TEMP = 'temp'


# (folder,        ext,  fps, rotate, max_sensor_val, scale, safe_name   )  
# ('Depth AHaT', 'pgm', 60,  None,   1055,           'log', 'depth_ahat')
sensors = [
    ('VLC RF','pgm',60, 'l', None, None, 'vlc_rf'),
    ('VLC LF','pgm',60, 'r', None, None, 'vlc_lf'),
    # ('VLC LR','pgm',30, 'r'),
    # ('VLC RR','pgm',30, 'r'),
    ('Depth Long Throw','pgm', 60, None, 7442, None, 'depth_long_throw'),
    # ('long_throw_reflectivity','pgm',30, None),
    ('PV','png',60, None, None, None, 'pv'),
    ('Depth AHaT','pgm', 60, None, 1055, None, 'depth_ahat'),
    ('REFLECTIVITY','pgm',60, None, 10922, 'log', 'reflectivity')
]



def process_videos(root_dir_path):
    print("PROCESS_VIDEOS: ", root_dir_path)

    # determing which kind of depth sensor was used
    depth_folder = None
    if os.path.exists(os.path.join(root_dir_path, 'Depth AHat')):
        depth_folder = 'Depth AHat'
    elif os.path.exists(os.path.join(root_dir_path, 'Depth Long Throw')):
        depth_folder = 'Depth Long Throw'

    # move reflectivity images and depth images into separate folders
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


    # make video output path
    if not os.path.isdir(os.path.join(root_dir_path, VIDEOS)):
        os.mkdir(os.path.join(root_dir_path, VIDEOS))

    # make temp path
    if not os.path.isdir(os.path.join(root_dir_path, TEMP)):
        os.mkdir(os.path.join(root_dir_path, TEMP))

    # initalize list to keep track of the min and max values for each sensor (to adjust the scaling)
    extremes = []
    for sensor_idx in range(len(sensors)):
        min_val = float('inf')
        max_val = -float('inf')
        extremes.append((min_val, max_val))



    for sensor_idx, sensor in enumerate(sensors):    
        name, filetype, fps, rotate, maxval, scale, safe_name = sensor
        min_val, max_val = extremes[sensor_idx]

        print(sensor_idx, name)

        # make temp path
        if not os.path.isdir(os.path.join(root_dir_path, TEMP, safe_name)):
            os.mkdir(os.path.join(root_dir_path, TEMP, safe_name))

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
            
            # get timestamps from filename
            timestamp = int(fname.split('\\')[-1].split('.')[0])       
            next_timestamp = int(next_fname.split('\\')[-1].split('.')[0])
            
            temp_file_path = os.path.join(root_dir_path, TEMP, safe_name, f"{i:05d}.png")
            file_list_path = temp_file_path.replace('\\', '/')

            # set first timestamp 
            if first_timestamp is None:
                first_timestamp = timestamp

                
            if True:
                # load frame (raw data)
                img = cv2.imread(fname, -1)

                # store min and max of raw data
                min_val = min(min_val, np.min(img))
                max_val = max(max_val, np.max(img))

                # scale raw image to better scale for visualizing
                if maxval is not None:
                    raw = cv2.imread(fname, -1) / maxval

                    if scale == 'log':
                        raw = np.log(raw*1000 + 1) / np.log(1001)

                    gray = (raw*255).astype('uint8')
                    img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                
                # otherwise, load non-raw data and let opencv handle it
                else:
                    img = cv2.imread(fname)


                # rotate
                if rotate == 'r':
                    img = imutils.rotate_bound(img, 90)
                if rotate == 'l':
                    img = imutils.rotate_bound(img, -90)


                # all frames except the first frame
                if previous_frame is not None:
                    next_frame = round((timestamp - first_timestamp) / round(10000000 / fps))

                    # if frame time was longer than output frame rate, pad output with previous frame
                    for i in range(0, (next_frame - previous_frame) - 1):
                        img_array.append(file_list_path)


                    img_array.append(file_list_path)
                    cv2.imwrite(temp_file_path, img)
                    previous_frame = next_frame
                
                # First frame
                else:
                    img_array.append(file_list_path)
                    cv2.imwrite(temp_file_path, img)
                    previous_frame = round((timestamp - first_timestamp) / round(10000000 / fps))
                # Image.open(f'./pgm/{name}/{f}').save(f'./png/{name}/image{i:05}.png')
                

        if len(img_array) > 0:
            out_str = ""
            for img_temp_path in img_array:
                out_str = f"{out_str}file {img_temp_path}\n"
            
            file_list_path = os.path.join(root_dir_path, TEMP, safe_name, "files.txt")

            with open(file_list_path, 'w') as f:
                f.write(out_str)

            output_video_path = os.path.join(root_dir_path, VIDEOS, f'{safe_name}.mp4')

            (
                ffmpeg
                .input(file_list_path, format='concat', safe=0, r=fps)
                .output(output_video_path, vcodec='libx264', pix_fmt='yuv420p', crf=23, r=fps)
                .run(overwrite_output=True)
            )

            # concat stereo views with
            # ffmpeg -r 60 -i depth_long_throw.mp4  -r 60 -i reflectivity.mp4 -filter_complex "hstack,format=yuv420p" -c:v libx264 -crf 18 -r 60 depth_reflectivity.mp4

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