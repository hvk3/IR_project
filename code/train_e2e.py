import models
import numpy as np
import simple_reader
import pickle
from pymongo import MongoClient
from gensim.models import doc2vec
from itertools import tee
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
from tensorflow.python.platform import flags
import keras

# Dynamic memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'numComments',
    3,
    'Number of comments to be considered while building and training model'
)
flags.DEFINE_integer('batchSize', 16, 'Batch size')
flags.DEFINE_integer('numEpochs', 50, 'Number of epochs')
flags.DEFINE_boolean(
    'add_batch_norm',
    False,
    'Add batch normalization to model?'
)
flags.DEFINE_boolean('add_dropout', False, 'Add dropout to model?')
flags.DEFINE_boolean('no_metadata', False, 'Ignore metadata in model?')
flags.DEFINE_boolean('no_audio', False, 'Ignore audio in model?')
flags.DEFINE_boolean('no_video', False, 'Ignore video in model?')
flags.DEFINE_boolean('no_description', False, 'Ignore description in model?')
flags.DEFINE_boolean('no_channel', False, 'Ignore channel name?')
flags.DEFINE_boolean(
    'vec2doc',
    False,
    'Train on text, instead of doc2vec features'
)


if __name__ == "__main__":
    embeddingSize = 400
    client = MongoClient()
    database = client.youtube8m
    collection = database.iteration3

    d2v = doc2vec.Doc2Vec.load('doc2vec_{}.model'.format(embeddingSize))
    max_len = vocab_size = 0
    # Load data generator
    dataGen = simple_reader.mongoDBgenerator(
        collection,
        d2v,
        FLAGS.numComments,
        1,
        FLAGS.batchSize,
        0.2,
        FLAGS.vec2doc
    )
    # Ensuring function calls within generator work
    _ = dataGen.next()
    if FLAGS.vec2doc:
        tokenizer = pickle.load(open('tokenizer.pickle', 'r'))
        # max_len = np.load('max_seq_length.npy').item()
        max_len = 15
        vocab_size = tokenizer.num_words
        del tokenizer
    model = models.model(
        embeddingSize,
        FLAGS.numComments,
        vocab_size,
        max_len,
        FLAGS.add_batch_norm,
        FLAGS.add_dropout,
        (not FLAGS.no_audio),
        (not FLAGS.no_video),
        (not FLAGS.no_metadata),
        (not FLAGS.no_channel),
        (not FLAGS.no_description)
    )
    # Visualize model
    plot_model(model, to_file='model_arch.png')
    # Split into train and validation generators
    dataGen1, dataGen2 = tee(dataGen)
    trainGen = simple_reader.getTrainValGens(dataGen1, True)
    valGen = simple_reader.getTrainValGens(dataGen2, False)
    # Load test generator
    testGen = simple_reader.mongoDBgenerator(
        collection,
        d2v,
        FLAGS.numComments,
        3,
        FLAGS.batchSize,
        0.2,
        FLAGS.vec2doc)
    # Train model
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=5,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        min_lr=0.01,
        verbose=1
    )
    # import pdb
    # pdb.set_trace()
    model.fit_generator(
        trainGen,
        validation_data=valGen,
        validation_steps=200,
        steps_per_epoch=1000,
        epochs=FLAGS.numEpochs,
        callbacks=[early_stop, reduce_lr]
    )
    # Evaluate model on test data
    metric_computed = 0.0
    num_batches = 0
    for testBatch in testGen:
        if testBatch:
            num_batches += 1
            # batch_metric_computed = model.evaluate(
            #     testBatch[0],
            #     testBatch[1],
            #     verbose=0
            # )
            prediction = model.predict(testBatch[0])
            for po in prediction:
                print np.max(po), np.argmax(po)
            # metric_computed += batch_metric_computed[0] / len(testBatch[1])
        else:
            break
    # if (FLAGS.vec2doc):
    #     metric_computed *= 100. / num_batches
    # print("Metric on test data:", metric_computed)

