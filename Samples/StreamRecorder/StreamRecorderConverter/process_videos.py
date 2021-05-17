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
from project_hand_eye_to_pv import project_hand_eye_to_pv
from utils import check_framerates, extract_tar_file
from convert_images import convert_images



OUTPUT_FRAME_RATE = 144 # Output Fps should be sufficiently high so that no sensor frames will need to be skipped.
                        # If the output fps is too slow, the video lengths will be slightly inconsistent.

REFLECTIVITY = 'REFLECTIVITY'
PLY = 'PLY'
VIDEOS = 'videos'
TEMP = 'processed_frames'


# (folder,         ext,   rotate,  max_sensor_val,  scale,     safe_name)  
# ('Depth AHaT',  'pgm',    None,            1055,  'log',  'depth_ahat')
sensors = [
    ('VLC RF',            'pgm',   'l',   None,   None,  'vlc_rf'          ),
    ('VLC LF',            'pgm',   'r',   None,   None,  'vlc_lf'          ),
    ('VLC LR',            'pgm',   'r',   None,   None,  'vlc_lr'          ),
    ('VLC RR',            'pgm',   'r',   None,   None,  'vlc_rr'          ),
    ('Depth Long Throw',  'pgm',  None,   7442,   None,  'depth_long_throw'),
    ('PV',                'png',  None,   None,   None,  'pv'              ),
    ('Depth AHaT',        'pgm',  None,   1055,   None,  'depth_ahat'      ),
    ('REFLECTIVITY',      'pgm',  None,  10922,  'log',  'reflectivity'    )
]


folder_names = [
    'VLC RF',
    'VLC LF',
    'VLC LR',
    'VLC RR',
    'Depth Long Throw',
    'PV',
    'Depth AHaT',
    'REFLECTIVITY',
]



def process_all_no_point_clouds(w_path, project_hand_eye=False):
    # Extract all tar
    for tar_fname in w_path.glob("*.tar"):
        print(f"Extracting {tar_fname}")
        tar_output = ''
        tar_output = w_path / Path(tar_fname.stem)
        tar_output.mkdir(exist_ok=True)
        extract_tar_file(tar_fname, tar_output)

    # Process PV if recorded
    if (w_path / "PV.tar").exists():
        # Convert images
        convert_images(w_path)

        # Project
        if project_hand_eye:
            project_hand_eye_to_pv(w_path)

    print("")
    check_framerates(w_path)

    process_videos(w_path)


# ffmpeg -r 60 -i depth_long_throw.mp4  -r 60 -i reflectivity.mp4 -filter_complex "hstack,format=yuv420p" -c:v libx264 -crf 18 -r 60 depth_reflectivity.mp4
# concat two videos side by side
def hstack_videos(root_dir_path, left_filename, right_filename, output_filename):
    lf_video = os.path.join(root_dir_path, VIDEOS, left_filename)
    rf_video = os.path.join(root_dir_path, VIDEOS, right_filename)
    stereo_vlc_front_output = os.path.join(root_dir_path, VIDEOS, output_filename)
    if os.path.exists(lf_video) and os.path.exists(rf_video):
        print(f"    Saving video: {output_filename}")
        (
            ffmpeg
            .input(lf_video, r=OUTPUT_FRAME_RATE)
            # .input(rf_video, r=OUTPUT_FRAME_RATE)
            .output(stereo_vlc_front_output, vcodec='libx264', filter_complex='hstack,format=yuv420p', crf=0, r=OUTPUT_FRAME_RATE)
            .global_args('-i', rf_video, '-loglevel', 'quiet')
            .run(overwrite_output=True)
        )

    else:
        print(f"    Skipping {output_filename}")


def process_videos(root_dir_path):
    # check if sensor folders exist, implying that they have already been extracted
    already_extracted = False
    for folder_name in folder_names:
        if os.path.exists(os.path.join(root_dir_path, folder_name)):
            already_extracted = True

    # if they don't exist, assume they weren't extracted yet. Perform the extraction
    if not already_extracted:
        print("Processing videos without point cloud reconstruction")
        process_all_no_point_clouds(root_dir_path)

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
        name, filetype, rotate, maxval, scale, safe_name = sensor

        if not os.path.isdir(os.path.join(root_dir_path, name)):
            print(f"{sensor_idx} {name} data not found. Skipping")
            continue
    
        print(sensor_idx, name)
        min_val, max_val = extremes[sensor_idx]

        # make temp path
        if not os.path.isdir(os.path.join(root_dir_path, TEMP, safe_name)):
            os.mkdir(os.path.join(root_dir_path, TEMP, safe_name))

        
        # this block loads all of the files and applies the appropriate adjustments to them (rotation and scaling)
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
                next_frame = round((timestamp - first_timestamp) / round(10000000 / OUTPUT_FRAME_RATE))

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
                previous_frame = round((timestamp - first_timestamp) / round(10000000 / OUTPUT_FRAME_RATE))
                
        
        # this block creates the video by writing the adjusted frames to a temporary folder and then creating a list of the frames for ffmpeg
        # Note that long frames are repeated multiple times in the list to match the output fps.
        if len(img_array) > 0:
            out_str = ""
            for img_temp_path in img_array:
                out_str = f"{out_str}file {img_temp_path}\n"
            
            file_list_path = os.path.join(root_dir_path, TEMP, safe_name, "files.txt")

            with open(file_list_path, 'w') as f:
                f.write(out_str)

            output_video_path = os.path.join(root_dir_path, VIDEOS, f'{safe_name}.mp4')
            print(f"    Saving video: {safe_name}.mp4")
            (
                ffmpeg
                .input(file_list_path, format='concat', safe=0, r=OUTPUT_FRAME_RATE)
                .output(output_video_path, vcodec='libx264', pix_fmt='yuv420p', crf=0, r=OUTPUT_FRAME_RATE)
                .global_args('-loglevel', 'quiet')
                .run(overwrite_output=True)
            )


        # print(sensor_idx)
        extremes[sensor_idx] = (min_val, max_val)


    print("* Creating side by side videos")
    # concat stereo views
    hstack_videos(root_dir_path, 'vlc_lf.mp4', 'vlc_rf.mp4', 'vlc_front_stero.mp4')
    hstack_videos(root_dir_path, 'depth_long_throw.mp4', 'reflectivity.mp4', 'depth_long_throw_and_reflectivity.mp4')
    hstack_videos(root_dir_path, 'depth_ahat.mp4', 'reflectivity.mp4', 'depth_ahat_and_reflectivity.mp4')


    for i in range(len(sensors)):
        print(f'{sensors[i][0]: <16}  min:{extremes[i][0]: >4}  max:{extremes[i][1]: >6}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process recorded data.')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")

    args = parser.parse_args()

    root_dir_path = Path(args.recording_path)
    process_videos(root_dir_path)