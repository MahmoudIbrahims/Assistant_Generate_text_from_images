from transformers import pipeline
import IPython.display as ipd


def text2speech(text):
  # Load the text-to-speech pipeline
  text_to_speech = pipeline("text-to-speech")
  # Generate speech
  audio = text_to_speech(text)
  return audio

#output.wav
def save_speech(audio,file_name):
  # Save the speech to a file
  with open(file_name, "wb") as f:
      f.write(audio["speech"])


def play_audio(file_name):
  return ipd.Audio(file_name, autoplay=True)




