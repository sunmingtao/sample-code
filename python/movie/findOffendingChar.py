base_file = 'MIAB-043~S-720'
base_folder = 'C:/Users/smt/Downloads/jap/'
input_video = base_folder + base_file + '.mp4'
output_video = base_folder + base_file + '_withSub.mp4'
output_audio = base_folder + base_file + '.mp3'
srt = base_folder + base_file + '.srt'

def inspect_offending_character(file_path, position):
    try:
        with open(file_path, 'rb') as file:  # Open the file in binary mode
            file.seek(position)  # Move the cursor to the position just before the error
            file.seek(max(0, position - 10), 0)  # Go back 10 bytes to get some context
            context = file.read(20)  # Read 20 bytes to get a good amount of surrounding text

        # Attempt to decode the context in UTF-8
        print("Context around the offending byte:")
        print(context.decode('utf-8'))

    except UnicodeDecodeError as e:
        print("Failed to decode the file in UTF-8. Here's the byte-level output for more context:")
        print(context)  # Show the raw bytes if decoding fails

    except Exception as e:
        print(f"An error occurred: {e}")


# Replace 'path_to_your_srt_file.srt' with the path to your SRT file
# Replace 820 with the exact byte position of your error if it differs
inspect_offending_character(srt, 515)
