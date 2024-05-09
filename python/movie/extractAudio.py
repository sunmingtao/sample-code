import subprocess

def extract_audio(input_video_path, output_audio_path):
    # Construct the ffmpeg command to extract audio
    command = [
        'ffmpeg',
        '-i', input_video_path,  # Input file path
        '-vn',                   # No video output
        '-acodec', 'libmp3lame', # Use MP3 codec
        '-q:a', '2',             # Quality level; lower numbers are better quality
        output_audio_path        # Output file path
    ]

    # Execute the command
    subprocess.run(command, check=True)

# Example usage
# input_video = 'C:/Users/smt/Downloads/jap/0419jap.mp4'
# output_audio = 'C:/Users/smt/Downloads/jap/0419jap.mp3'
# extract_audio(input_video, output_audio)
