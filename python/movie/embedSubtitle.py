from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip

def make_subtitle_clip(srt_file):
    """ Generate a subtitle clip """
    def generator(txt):
        """ Generates a text clip for subtitles. """
        # You can customize the appearance here
        return TextClip(txt, font='Arial', fontsize=60, color='yellow')

    return SubtitlesClip(srt_file, generator)

def add_subtitles_to_video(video_file, srt_file, output_file):
    """ Add subtitles to a video using the SRT file """
    video = VideoFileClip(video_file)
    subtitles = make_subtitle_clip(srt_file)

    # Overlay the subtitles on the original video
    final = CompositeVideoClip([video, subtitles.set_position(('center','bottom'))])

    # Write the result to a file
    final.write_videofile(output_file, codec='libx264', temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')

# if __name__ == "__main__":
#     add_subtitles_to_video("C:/Users/smt/Downloads/jap/0419jap.mp4", "C:/Users/smt/Downloads/jap/0419jap.srt", "C:/Users/smt/Downloads/jap/0419jap_withSub.mp4")
