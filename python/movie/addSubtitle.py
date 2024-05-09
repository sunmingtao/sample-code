from extractAudio import *
from translateJapAudio import *
from embedSubtitle import *
import time
from datetime import timedelta

base_file = 'FJL-006~S'
base_folder = 'C:/Users/smt/Downloads/jap/'
input_video = base_folder + base_file + '.mp4'
output_video = base_folder + base_file + '_withSub.mp4'
output_audio = base_folder + base_file + '.mp3'
srt = base_folder + base_file + '.srt'

start_time = time.time()  # Get the start time

print("Extract audio")
extract_audio(input_video, output_audio)
print("Transcribe video")
translate_audio(output_audio, "medium", "Japanese", srt)
# print("Embed subtitle")
# add_subtitles_to_video(input_video, srt, output_video)

end_time = time.time()  # Get the end time

duration = end_time - start_time  # Calculate the duration in seconds
formatted_duration = str(timedelta(seconds=round(duration)))
print(f"The function took {formatted_duration} to run.")
