import whisper

def format_srt(segments):
    srt_content = []
    for index, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text']
        srt_content.append(f"{index}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_content)

def format_timestamp(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def translate_audio(file_path, model_type="small", source_language="Japanese", output_file="C:/Users/smt/Downloads/jap/0419jap.srt"):
    # Load the model
    model = whisper.load_model(model_type)

    # Load and process the audio file
    result = model.transcribe(file_path, language=source_language)

    # Generate SRT content
    srt_content = format_srt(result["segments"])

    # Save to a file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(srt_content)

    print(f"Transcribe and subtitles saved to {output_file}")

# if __name__ == "__main__":
#     translate_audio("C:/Users/smt/Downloads/jap/0419jap.mp3")
