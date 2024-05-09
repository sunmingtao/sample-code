import chardet

base_file = 'JUQ-426~S-720'
base_folder = 'C:/Users/smt/Downloads/jap/'
input_video = base_folder + base_file + '.mp4'
output_video = base_folder + base_file + '_withSub.mp4'
output_audio = base_folder + base_file + '.mp3'
srt = base_folder + base_file + '.srt'

with open(srt, 'rb') as f:
    result = chardet.detect(f.read())  # Detect file encoding
encoding = result['encoding']
print (encoding)