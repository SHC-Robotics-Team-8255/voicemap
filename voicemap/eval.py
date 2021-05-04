import soundfile as sf
import numpy as np
import time
from models import get_baseline_convolutional_encoder, build_siamese_net
from IPython.display import Audio, clear_output, display, SVG
import sounddevice as sd
from sklearn.manifold import TSNE, MDS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import model_to_dot
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

#%%

import sys
sys.path.append('../')
from config import PATH, LIBRISPEECH_SAMPLING_RATE
from librispeech import LibriSpeechDataset

#%% md

model_path = '../models/n_seconds/siamese__nseconds_3.0__filters_32__embed_64__drop_0.05__r_0.hdf5'
downsampling = 4
n_seconds = 3
filters = 128
embedding_dimension = 64
dropout = 0.0
input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)


#%%
encoder = get_baseline_convolutional_encoder(filters, embedding_dimension, dropout=dropout)
siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
opt = Adam(clipnorm=1.)
#siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
siamese.load_weights(model_path)

#%%

# SVG(model_to_dot(siamese, show_shapes=True).create(prog='dot', format='svg'))

#%%

validation_set = ['train-clean-100','train-clean-360','dev-clean']
n_seconds = 3
n_shot_classification = 1
k_way_classification = 5

num_tasks = 10

#%% md

# Get data

#%%

valid_sequence = LibriSpeechDataset(validation_set, n_seconds, stochastic=False)

#%% md

# Evaluation loop

#%%

name = input('Enter your name: ')

correct = []
answers = []
for i in range(num_tasks):
    print('******* Trial {} of {} ******'.format(i+1, num_tasks))
    query_sample, support_set_samples = valid_sequence.build_n_shot_task(
        k_way_classification, n_shot_classification)

    query_audio = Audio(data=query_sample[0], rate=LIBRISPEECH_SAMPLING_RATE)
    # sound = AudioSegment.from(query_sample[0])
    # play(query_sample[0])


    print('Match this sample:')
    sd.play(query_sample[0], LIBRISPEECH_SAMPLING_RATE)
    sd.wait()
    #display(query_audio)

    support_set_audio = [
        (i+1, support_set_samples[0][i, :]) for i in range(k_way_classification)]
    support_set_names = [
        valid_sequence.df[valid_sequence.df['speaker_id']==i]['name'].values[0] for i in support_set_samples[1]]

    # Index, name, audio
    support_set = list(zip(list(zip(*support_set_audio))[0], support_set_names, list(zip(*support_set_audio))[1]))

    # Shuffle and record correct answer
    np.random.shuffle(support_set)
    correct.append(list(zip(*support_set))[0].index(1) + 1)
    support_set_audio = list(zip(*support_set))[2]
    support_set_names = list(zip(*support_set))[1]

    print('To one of these 5 speakers:')
    for i, audio in enumerate(support_set_audio):
        print('{}: {}'.format(i+1, support_set_names[i]))
        sd.play(audio, LIBRISPEECH_SAMPLING_RATE)
        sd.wait()
        #display(audio)

    time.sleep(0.01)
    while True:
        answer = input('Enter correct speaker number: ')

        if answer in ('1','2','3','4','5'):
            break
        else:
            print('Typo!')

    answers.append(int(answer))

    print('The correct answer was {}'.format(correct[-1]))

    _ = input('Press any key to continue...')

    clear_output()


num_correct = sum(a == c for a, c in zip(answers, correct))
with open(PATH + '/data/human_evaluation.csv', 'a') as f:
    print('{},{},{}'.format(name, num_correct, num_tasks))
print('You got {} out {} correct!'.format(num_correct, num_tasks))

