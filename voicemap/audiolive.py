import sounddevice as sd
import numpy as np
import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model("model.h5")



def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


fs = 16000
duration = 1  # seconds
recording = None
while True:
    newrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    if recording is not None:
        recording = audio_to_fft([recording])
        prediction = model.predict(recording)
        print(prediction)
        prediction = np.argmax(prediction, axis=-1)
        print(prediction)
    sd.wait()
    recording = newrecording
