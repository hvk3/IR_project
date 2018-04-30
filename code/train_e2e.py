import models
import numpy as np
import simple_reader
import pickle
from pymongo import MongoClient
from gensim.models import doc2vec
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

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
flags.DEFINE_boolean('test_demo', False, 'Test model on some examples?')
flags.DEFINE_integer(
    'numComments',
    3,
    'Number of comments to be considered while building and training model'
)
flags.DEFINE_integer('batchSize', 16, 'Batch size')
flags.DEFINE_integer('numEpochs', 3, 'Number of epochs')
flags.DEFINE_integer('maxLen', 5, 'Max length for padding')
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


def check_labels(y_train):
    for i in xrange(len(y_train)):
        print map(lambda x: np.argmax(x), y_train[i])


if __name__ == "__main__":
    embeddingSize = 400
    client = MongoClient()
    database = client.youtube8m
    collection = database.iteration6

    d2v = doc2vec.Doc2Vec.load('doc2vec_{}.model'.format(embeddingSize))
    vocab_size = 0
    max_len = 0
    if (FLAGS.vec2doc):
        max_len = FLAGS.maxLen
    # Load data generator
    dataGen = simple_reader.mongoDBgenerator(
        collection,
        d2v,
        FLAGS.numComments,
        1,
        FLAGS.batchSize,
        0.2,
        FLAGS.vec2doc,
        max_len
    )
    # Ensure tokenizer works
    _ = dataGen.next()
    if FLAGS.vec2doc:
        tokenizer = pickle.load(open('tokenizer.pickle', 'r'))
        # max_len = np.load('max_seq_length.npy').item()
        vocab_size = tokenizer.num_words
        del tokenizer
    # Ensuring function calls within generator work
    _ = dataGen.next()
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
    # Load test generator
    testGen = simple_reader.mongoDBgenerator(
        collection,
        d2v,
        FLAGS.numComments,
        3,
        FLAGS.batchSize,
        0.2,
        FLAGS.vec2doc,
        max_len
    )
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
    if FLAGS.test_demo:
        model = load_model("IR_demo_project.h5")
    else:
        models.customTrain(
            model,
            dataGen,
            FLAGS.numEpochs,
            FLAGS.batchSize,
            valRatio=0.2,
            no_acc=True
        )
        model.save("IR_demo_project.h5")
    # Evaluate model on test data
    from scipy import spatial
    metric_computed = 0.0
    num_batches = 0
    for tb in testGen:
        if tb:
            num_batches += 1
            testBatch, chak = tb
            batch_metric_computed = model.evaluate(
                 testBatch[0],
                 testBatch[1],
                 verbose=0
            )
            prediction = model.predict(testBatch[0])
            #for po in prediction:
            #    print np.max(po), np.argmax(po)
            metric_computed += batch_metric_computed / len(testBatch[1])
            print chak
            testvec = d2v.infer_vector(chak[1])
            print("Cosine Similarity Score For Original Title: %f", 50 * (2 - spatial.distance.cosine(testvec, prediction[-1])))
            suggested_title = raw_input("Enter suggested title: ")
            testvec = d2v.infer_vector(suggested_title)
            print("Cosine Similarity Score : %f", 50 * (2 - spatial.distance.cosine(testvec, prediction[-1])))
        else:
            break
    # if (FLAGS.vec2doc):
    #     metric_computed *= 100. / num_batches
    print("Metric on test data:", metric_computed)

