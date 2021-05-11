from pydub import AudioSegment as am
import os

for root, dirs, files in os.walk('../data/LibriSpeech/roboticsclub'):
    for name in files:
        randomfile=os.path.join(root, name)
        sound = am.from_file(randomfile, format='flac')
        sound = sound.set_frame_rate(16000)
        sound.export(randomfile.replace(".flac", ".wav"), format='wav')

