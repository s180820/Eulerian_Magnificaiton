# script to compress videofiles in a directory

import os
import subprocess
import sys

def cut_video(video_file, start_time, end_time, output_file):
    command = "ffmpeg -i {} -ss {} -to {} -c copy {}".format(
        video_file, start_time, end_time, output_file)
    subprocess.call(command, shell=True)

def compress_video(video_file, output_file):
    #compress mp4 video to H.263, H.264 and H.265
    command = "ffmpeg -i {} -vcodec h264 -acodec mp2 {}".format(
        video_file, output_file)
    subprocess.call(command, shell=True)

# cut video into 70 random 10 second clips
def cut_video_into_clips(video_file, output_dir):
    command = "ffmpeg -i {} -c copy -map 0 -segment_time 10 -f segment -reset_timestamps 1 {}/%03d.mp4".format(
        video_file, output_dir)
    subprocess.call(command, shell=True)

# compress all videos in a directory
def compress_videos_in_dir(dir, output_dir):
    for filename in os.listdir(dir):
        if filename.endswith(".mp4"):
            video_file = os.path.join(dir, filename)
            output_file = os.path.join(output_dir, filename)
            compress_video(video_file, output_file)

#extract compression bit rate label and turn it into a one hot vector of size 5 as tensor 
def extract_bitrate(video_file):
    bitrate = subprocess.check_output("ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 {}".format(video_file), shell=True)
    bitrate = int(bitrate)
    if bitrate < 1000000:
        return [1, 0, 0, 0, 0]
    elif bitrate < 2000000:
        return [0, 1, 0, 0, 0]
    elif bitrate < 3000000:
        return [0, 0, 1, 0, 0]
    elif bitrate < 4000000:
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compress.py <video_file> <output_file>")
        exit(1)
    video_file = sys.argv[1]
    output_dir = sys.argv[2]
    #initialise directories
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(output_dir + "/clips")
        os.mkdir(output_dir + "/compressed")
    #cut video into 10 second clips
    cut_video_into_clips(video_file, output_dir + "/clips")
    #compress videos in directory
    compress_videos_in_dir(output_dir + "/clips", output_dir + "/compressed")
