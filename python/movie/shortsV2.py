from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from common import *
from PIL import Image as pil
from pkg_resources import parse_version

if parse_version(pil.__version__)>=parse_version('10.0.0'):
    pil.ANTIALIAS=pil.LANCZOS

coordinate_map = {
    "L": (10, 290, 960, 1070),
    "R": (970, 290, 1910, 1070)
}

def parse_string(input_string):
    # Split the input string by spaces
    tokens = input_string.split()

    # Initialize variables to store the 2nd and 4th tokens
    leftOrRight = None

    # Check if there are at least four tokens啊
    if len(tokens) == 5:
        date = tokens[0]
        leftOrRight = tokens[1]
        hand = tokens[2]
        startTime, endTime = parse_time(tokens[3])
        comment = tokens[4]
    else:
        raise ValueError("missing some params")

    return date, leftOrRight, hand, time_str_to_seconds(startTime), time_str_to_seconds(endTime), comment

def get_short_raw(param):
    date = param[0]
    clip_raw = VideoFileClip("C:/Users/smt/Videos/" + date + "-Zoom-NL100.mkv").subclip(param[3], param[4])
    hand_coordinates = coordinate_map[param[1]]
    clip_hand = clip_raw.crop(hand_coordinates[0], hand_coordinates[1], hand_coordinates[2], hand_coordinates[3])
    new_width = 600  # New width in pixels
    # new_height = 720  # New height in pixels
    # Resize the video to the new aspect ratio
    clip_hand = clip_hand.resize(width=new_width)
    me_coordinates = 1350, 0, clip_raw.size[0]-150, 270
    clip_me = clip_raw.crop(me_coordinates[0], me_coordinates[1], me_coordinates[2], me_coordinates[3])
    clip_me = clip_me.resize(width=new_width)
    total_height = clip_me.size[1] + clip_hand.size[1]
    stacked_video = CompositeVideoClip([clip_hand.set_position(('center', 0)),
                                  clip_me.set_position(('center', clip_hand.size[1]-80))],
                                 size=(clip_hand.size[0], total_height))
    image_clip = ImageClip("YH3333poster.png", duration=stacked_video.duration)
    image_clip_resized = image_clip.resize(width=600)
    total_height += image_clip.size[1]
    composite_clip = CompositeVideoClip([stacked_video.set_position(('center', image_clip_resized.size[1] + 50)),
                                         image_clip_resized.set_position(('center', 50))],
                                 size=(image_clip_resized.size[0], total_height))
    fileName = date+"_"+param[1]+"_"+param[2]+"_"+param[5]+".mp4"
    return fileName, composite_clip

with open('shorts.txt', 'r', encoding='utf-8') as file:
    # Loop through each line in the file
    for line in file:
        fileName, clip_final = get_short_raw(parse_string(line.strip()))
        clip_final.write_videofile(fileName, codec="libx264")
