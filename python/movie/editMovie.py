from moviepy.editor import *
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout

coordinate_map = {
    "L": (100, 100),
    "R": (1100, 100)
}


def parse_string(input_string):
    # Split the input string by spaces
    tokens = input_string.split()

    # Initialize variables to store the 2nd and 4th tokens
    leftOrRight = None

    # Check if there are at least four tokens
    if len(tokens) >= 4:
        # Assign the 2nd and 4th tokens to their respective variables
        leftOrRight = tokens[1]
        startTime, endTime = parse_time(tokens[3])

    return leftOrRight, time_str_to_seconds(startTime), time_str_to_seconds(endTime)

def parse_time(input_string):
    # Split the input string by "-"
    tokens = input_string.split("-")
    # Initialize variables to store the first and second tokens
    first_token = None
    second_token = None

    # Check if there are at least two tokens
    if len(tokens) >= 2:
        # Assign the first and second tokens to their respective variables
        first_token = tokens[0]
        second_token = tokens[1]

    return first_token, second_token

def time_str_to_seconds(time_str):
    try:
        if len(time_str) == 6:
            hours = int(time_str[:2])
            minutes = int(time_str[2:4])
            seconds = int(time_str[4:])
            total_seconds = (hours * 3600) + (minutes * 60) + seconds
        elif len(time_str) == 4:
            minutes = int(time_str[:2])
            seconds = int(time_str[2:])
            total_seconds = (minutes * 60) + seconds
        else:
            raise ValueError("Input string should be in 'hhmmss' or 'mmss' format.")

        return total_seconds
    except ValueError as e:
        return str(e)

def process_line(line):
    leftOrRight, startTime, endTime = parse_string(line)
    clip = clip1.subclip(startTime, endTime).fx(fadeout, 2)
    maskClip = clip2.set_position(coordinate_map[leftOrRight]).loop().set_duration(3)
    return CompositeVideoClip([clip, maskClip])

basePath = "C:/Users/smt/Videos/"

clip2 = VideoFileClip("yellow.gif", has_mask=True)

highlight_clips = []

inputFile = None

with open('hilights.txt', 'r') as file:
    inputFile = file.readline().strip()
    clip1 = VideoFileClip(basePath + inputFile + "-Zoom-NL100.mkv")
    # Loop through each line in the file
    for line in file:
        # Process each line as needed
        highlight_clips.append(process_line(line))

final_clip = concatenate_videoclips(highlight_clips)
final_clip.write_videofile(basePath + inputFile + "-raw.mp4")

