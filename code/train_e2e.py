import models
import simple_reader
from pymongo import MongoClient
from gensim.models import doc2vec
from itertools import tee
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
flags.DEFINE_boolean('no_metadata', False, 'Ignore metadata in model?')
flags.DEFINE_boolean('no_audio', False, 'Ignore audio in model?')
flags.DEFINE_boolean('no_video', False, 'Ignore video in model?')
flags.DEFINE_boolean('no_description', False, 'Ignore description in model?')
flags.DEFINE_boolean('no_channel', False, 'Ignore channel name?')
flags.DEFINE_boolean('vec2doc', False, 'Train on text, instead of doc2vec features')


if __name__ == "__main__":
	embeddingSize = 100
	# Define model
	if FLAGS.no_metadata:
		model = models.no_metadata_model(embeddingSize, FLAGS.numComments, FLAGS.add_batch_norm, FLAGS.add_dropout)
	elif FLAGS.no_audio:
		model = models.no_audio_model(embeddingSize, FLAGS.numComments, FLAGS.add_batch_norm, FLAGS.add_dropout)
	elif FLAGS.no_video:
		model = models.no_video_model(embeddingSize, FLAGS.numComments, FLAGS.add_batch_norm, FLAGS.add_dropout)
	elif FLAGS.no_description:
		model = models.no_description_model(embeddingSize, FLAGS.numComments, FLAGS.add_batch_norm, FLAGS.add_dropout)
	elif FLAGS.no_channel:
		model = models.no_channelname_model(embeddingSize, FLAGS.numComments, FLAGS.add_batch_norm, FLAGS.add_dropout)
	elif FLAGS.numComments == 0:
		model = models.no_comments_model(embeddingSize, FLAGS.add_batch_norm, FLAGS.add_dropout)
	elif FLAGS.vec2doc:
		print("Sorry, not implemented yet!")
	else:
		model = models.no_sent2vec_model(embeddingSize, FLAGS.numComments, FLAGS.add_batch_norm, FLAGS.add_dropout)
	# Visualize model
	plot_model(model, to_file='model_arch.png')
	client = MongoClient()
	database = client.youtube8m
	collection = database.iteration3
	d2v = doc2vec.Doc2Vec.load('doc2vec_100.model')
	# Load data generator
	dataGen = simple_reader.mongoDBgenerator(collection, d2v, FLAGS.numComments, 1, FLAGS.batchSize)
	# Split into train and validation generators
	dataGen1, dataGen2 = tee(dataGen)
	trainGen, valGen = simple_reader.getTrainValGens(dataGen1, True), simple_reader.getTrainValGens(dataGen2, False)	
	# Load test generator
	testGen = simple_reader.mongoDBgenerator(collection, d2v, FLAGS.numComments, 3, FLAGS.batchSize, 0.2,
			use_audio=(FLAGS.no_audio==False),
			use_video=(FLAGS.no_video==False),
			use_desc=(FLAGS.no_description==False),
			use_metadata=(FLAGS.no_metadata==False),
			use_comments = (FLAGS.numComments > 0),
			use_channel = (FLAGS.no_channel==False))
	# Train model
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.01, verbose=1)
	model.fit_generator(trainGen, validation_data=valGen, validation_steps=200, steps_per_epoch=1000, epochs=FLAGS.numEpochs, callbacks=[early_stop, reduce_lr])
	# Ealuate model on test data
	finalLoss = 0.0
	for testBatch in testGen:
		if testBatch:
			finalLoss += model.evaluate(testBatch[0], testBatch[1], verbose=0) / len(testBatch[1])
		else:
			break
	print("Loss on test data:", finalLoss)
