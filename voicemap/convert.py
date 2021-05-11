from pydub import AudioSegment
import os
#name = "anson"

# audio = AudioSegment.from_file(f"data/LibriSpeech/roboticsclubval/3/{name}2.flac")
# audio.export("data/marina1.flac", format="flac") # Exports to a wav file in the current path.

# exit()
name = "../data/LibriSpeech/roboticsclub/3/anson2.flac"
audio = AudioSegment.from_file(name)
for parts in range(0, len(audio), 1000):
    t1 = parts
    t3 = parts + 1000
    newAudio = audio[t1:t3]
    newAudio = newAudio.set_frame_rate(16000)
    newAudio.export(name.replace('.flac', f'2-{parts}.wav'), format="wav")
exit()
# for parts in range(0, len(audio), 4000):
for root, dirs, files in os.walk('../data/LibriSpeech/roboticsclub'):
    for name in files:
        if not name.endswith(".wav"): continue
        name = os.path.join(root, name)
        audio = AudioSegment.from_file(name)
        for parts in range(0, len(audio), 1000):
            t1 = parts
            t3 = parts + 1000
            newAudio = audio[t1:t3]
            newAudio.export(name.replace('.wav', f'-{parts}.wav'), format="wav")
