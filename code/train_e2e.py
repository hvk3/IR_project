import models
import simple_reader
from pymongo import MongoClient
from gensim.models import doc2vec
from keras.utils import plot_model
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import flags
import keras

# Dynamic memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('numComments', 3, 'Number of comments ot be consideredwhile building and training model')
flags.DEFINE_integer('batchSize', 16, 'Batch size')
flags.DEFINE_integer('numEpochs', 50, 'Number of epochs')
flags.DEFINE_boolean('add_batch_norm', False, 'Add batch normalization to model?')
flags.DEFINE_boolean('add_dropout', False, 'Add dropout to model?')


if __name__ == "__main__":
	embeddingSize = 100
	# Define model
	model = models.no_sent2vec_training_model(embeddingSize, FLAGS.numComments, FLAGS.add_batch_norm, FLAGS.add_dropout)
	# Visualize model
	plot_model(model, to_file='model_arch.png')
	client = MongoClient()
	database = client.youtube8m
	collection = database.iteration2
	d2v = doc2vec.Doc2Vec.load('doc2vec_100.model')
	# Load generator
	generator = simple_reader.mongoDBgenerator(collection, d2v, FLAGS.numComments, 1, FLAGS.batchSize)
	# Train model
	model.fit_generator(generator, steps_per_epoch=1000, epochs=FLAGS.numEpochs)

