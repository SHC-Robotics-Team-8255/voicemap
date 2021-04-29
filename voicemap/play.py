from pydub import AudioSegment
from pydub.playback import play
import os, random

n=0
# random.seed();
randomfile=None
for root, dirs, files in os.walk('D:/audio/LibriSpeech/'):
  for name in files:
    n=n+1
    if random.uniform(0, n) < 1 and name.endswith(".flac"): 
        randomfile=os.path.join(root, name)

print(randomfile)
sound = AudioSegment.from_file(randomfile)
play(sound)