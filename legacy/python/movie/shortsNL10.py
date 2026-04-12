from moviepy.editor import *
from common import *
from PIL import Image as pil
from pkg_resources import parse_version

if parse_version(pil.__version__)>=parse_version('10.0.0'):
    pil.ANTIALIAS=pil.LANCZOS

coordinate_map = {
    "L": (15, 15, 960, 800),
    "R": (970, 0, 1900, 800)
}

me_coordinates = 1350, 740, 1900, 1070

def parse_string(input_string):
    # Split the input string by spaces
    tokens = input_string.split()

    # Initialize variables to store the 2nd and 4th tokens
    leftOrRight = None

    # Check if there are at least four tokens
    if len(tokens) >= 5:
        date = tokens[0]
        leftOrRight = tokens[1]
        hand = tokens[2]
        startTime, endTime = parse_time(tokens[3])
        comment = tokens[4]

    return date, leftOrRight, hand, time_str_to_seconds(startTime), time_str_to_seconds(endTime), comment

def get_short_raw(param):
    date = param[0]
    clip_raw = VideoFileClip("resources/nl10/" + date + ".mkv").subclip(param[3], param[4])
    print(clip_raw.size)
    hand_coordinates = coordinate_map[param[1]]
    clip_hand = clip_raw.crop(hand_coordinates[0], hand_coordinates[1], hand_coordinates[2], hand_coordinates[3])
    new_width = 600  # New width in pixels
    new_height = 720  # New height in pixels
    # Resize the video to the new aspect ratio
    clip_hand = clip_hand.resize((new_width, new_height))
    clip_me = clip_raw.crop(me_coordinates[0], me_coordinates[1], me_coordinates[2], me_coordinates[3])
    clip_me = clip_me.resize(width=new_width)
    total_height = clip_me.size[1] + clip_hand.size[1]
    stacked_video = CompositeVideoClip([clip_hand.set_position(('center', 0)),
                                  clip_me.set_position(('center', clip_hand.size[1]-100))],
                                 size=(clip_hand.size[0], total_height))
    fileName = date+"_"+param[1]+"_"+param[2]+"_"+param[5]+".mp4"
    return fileName, stacked_video

with open('shorts.txt', 'r', encoding='utf-8') as file:
    # Loop through each line in the file
    for line in file:
        fileName, clip_final = get_short_raw(parse_string(line))
        clip_final.write_videofile(fileName, codec="libx264")
