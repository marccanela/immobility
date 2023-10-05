"""
Version: 05/07/2023
@author: mcanela
VIDEO CORRECTOR FOR POLYBOX MISPLACEMENT OF FRAMES
"""

import os
import numpy as np
from pims import Video
import cv2
import shutil

directory = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-09 - Young males/Python Immobility/Data/corrected videos/'

def get_events(dat_path):
    with open(dat_path, 'r') as fp:
        n_lines_header = 0
        for line in fp:
            n_lines_header += 1
            line = line.strip()
            if line == '':
                break
    events = np.genfromtxt(dat_path, delimiter='\t', skip_header=n_lines_header, dtype=int)
    return events

def read_rec_duration(events, event_num=11):
    eleven = events[events[:, 1] == event_num]
    last_ts = eleven[-2:, 0]
    is_ttl = np.any(events[:, 1] == 15)  # TTL event should be type 15
    return last_ts, is_ttl


def add_blank_frames(video_path, missing_frames):
    # Create a temporary file to store the modified video
    temp_output_path = video_path[:-4] + "_temp.mp4"

    # Read the video
    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create blank frames
    blank_frame = np.zeros((frame_height, frame_width, 3), np.uint8)  # Assuming 3 channels for color

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use "mp4v" codec for MP4 format
    output_video = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    # Add blank frames at the beginning
    for _ in range(missing_frames):
        output_video.write(blank_frame)

    # Write remaining original video frames
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = video.read()
        if not ret:
            break
        output_video.write(frame)
        frame_count += 1

    # Release resources
    video.release()
    output_video.release()

    # Replace the original video with the modified video
    shutil.move(temp_output_path, video_path)

    print(f"Modified video saved at {video_path} with {missing_frames} blank frames at the beginning.")


for filename in os.listdir(directory):
    if filename.endswith('.dat'):
        tag = filename[:-4]
        
        video_tag = tag + '_1.avi'
        video_path = os.path.join(directory, video_tag)

        v = Video(video_path)
        frame_rate = int(v.frame_rate)
        n_frames = len(v)     
        v.close()
        
        dat_tag = tag + '.dat'
        dat_path = os.path.join(directory, dat_tag)
    
        events = get_events(dat_path)
        last_ts, is_ttl = read_rec_duration(events)
        missing_frames = int((last_ts[0] / 1000) * frame_rate - n_frames)
        
        add_blank_frames(video_path, missing_frames)
    






