import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import multiprocessing

from utils import preprocess_instances, NShotEvaluationCallback, BatchPreProcessor
from models import get_baseline_convolutional_encoder, build_siamese_net
from librispeech import LibriSpeechDataset
from config import LIBRISPEECH_SAMPLING_RATE, PATH


# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##############
# Parameters #
##############
n_seconds = 3
downsampling = 4
batchsize = 64
filters = 128
embedding_dimension = 64
dropout = 0.0
training_set = ['roboticsclub']
validation_set = 'roboticsclubval'
pad = True
num_epochs = 50
evaluate_every_n_batches = 500
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5

# Derived parameters
input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)
param_str = 'siamese__filters_{}__embed_{}__drop_{}__pad={}'.format(filters, embedding_dimension, dropout, pad)


###################
# Create datasets #
###################
train = LibriSpeechDataset(training_set, n_seconds, pad=pad)
valid = LibriSpeechDataset(validation_set, n_seconds, stochastic=False, pad=pad)

batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))
train_generator = (batch_preprocessor(batch) for batch in train.yield_verification_batches(batchsize))
valid_generator = (batch_preprocessor(batch) for batch in valid.yield_verification_batches(batchsize))


################
# Define model #
################
encoder = get_baseline_convolutional_encoder(filters, embedding_dimension, dropout=dropout)
siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
opt = Adam(clipnorm=1.)
siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# plot_model(siamese, show_shapes=True, to_file=PATH + '/plots/siamese.png')
print(siamese.summary())


#################
# Training Loop #
#################

siamese.fit(
    x=train_generator,
    steps_per_epoch=evaluate_every_n_batches,
    validation_data=valid_generator,
    validation_steps=100,
    epochs=num_epochs,
    # workers=multiprocessing.cpu_count(),
    use_multiprocessing=False,
    callbacks=[
        # First generate custom n-shot classification metric
        NShotEvaluationCallback(
            num_evaluation_tasks, n_shot_classification, k_way_classification, valid,
            preprocessor=batch_preprocessor,
        ),
        # Then log and checkpoint
        CSVLogger(PATH + '/logs/{}.csv'.format(param_str)),
        ModelCheckpoint(
            PATH + '/models/{}.hdf5'.format(param_str),
            monitor='val_{}-shot_acc'.format(n_shot_classification),
            mode='max',
            save_best_only=True,
            verbose=True
        ),
        ReduceLROnPlateau(
            monitor='val_{}-shot_acc'.format(n_shot_classification),
            mode='max',
            verbose=1
        )
    ]
)
