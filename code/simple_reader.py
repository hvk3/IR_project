import json
import numpy as np
import os
import base64
import tensorflow as tf
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
from google.protobuf.json_format import MessageToJson
from gensim.models import doc2vec


# For preparing text for LSTM model, use this
# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/

def get_titles(collection):
	titles = []
	vocab = {}
	count = 0
	for record in collection.find({}):
		title = record['metadata']['metadata']['title']
		for word in title.split(' '):
			word = word.lower().strip()
			if (word not in vocab):
				vocab[word] = count
				count += 1
		titles.append(map(lambda x: vocab[x.lower().strip()], title.split(' ')))
	np.save('titles.npy', titles)
	np.save('vocab.npy', vocab)
	return titles


def extract_relevant_info(json_parsed_record, which_generator):
	video_id = base64.b64decode(json_parsed_record['features']['feature']['video_id']['bytesList']['value'][0])
	try:
		labels = json_parsed_record['features']['feature']['labels']['int64List']['value']
	except:
		labels = []
	mean_audio = json_parsed_record['features']['feature']['mean_audio']['floatList']['value']
	mean_rgb = json_parsed_record['features']['feature']['mean_rgb']['floatList']['value']
	parsed_record = {
		"video_id": video_id
		,"labels": labels
		,"mean_audio": mean_audio
		,"mean_rgb": mean_rgb
		,"which_generator": which_generator
	}
	return parsed_record


def generator(location, which_generator, batch_size):
	protobuf_records = filter(lambda x: 'tfrecord' in x, os.listdir(location))
	if (which_generator == 1):
		protobuf_records = filter(lambda x: 'train' in x, protobuf_records)
	elif (which_generator == 2):
		protobuf_records = filter(lambda x: 'test' in x, protobuf_records)
	elif (which_generator == 3):
		protobuf_records = filter(lambda x: 'validate' in x, protobuf_records)
	protobuf_records = map(lambda x: os.path.join(location, x), protobuf_records)
	i = 0
	for j in range(len(protobuf_records) // batch_size):
		print(j, "out of", len(protobuf_records) // batch_size, "done")
		i %= batch_size
		json_parsed_records = []
		for record in protobuf_records[i * batch_size : (i + 1) * batch_size]:
			for record_ in tf.python_io.tf_record_iterator(record):
				json_parsed_record = json.loads(MessageToJson(tf.train.Example.FromString(record_)))
				json_parsed_records.append(extract_relevant_info(json_parsed_record, which_generator))
		i += 1
		if (len(json_parsed_records) == 0):
			continue
		yield np.array(json_parsed_records)


def rankAndSelectComments(comments, select):
	likes = []
	for comment in comments:
		likes.append(int(comment['likes']))
	sortedByLikes = np.argsort(likes)[::-1]
	selectedComments = []
	for i in range(len(sortedByLikes)):
		selectedComments.append(comments[sortedByLikes[i]]['comment'])
	return selectedComments[:select]


def filled_metadata(metadata):
	commentCount =  metadata.get('commentCount', 0)
	viewCount = metadata.get('viewCount', 0)
	favoriteCount =  metadata.get('favoriteCount', 0)
	dislikeCount = metadata.get('dislikeCount', 0)
	likeCount = metadata.get('likeCount', 0)
	return [commentCount, viewCount, favoriteCount, dislikeCount, likeCount]


def mongoDBgenerator(collection, d2vmodel, numComments, which, batch_size, validation_ratio, use_audio, use_video
		, use_desc, use_metadata, use_comments, use_channel):
	print("Found", collection.find({"which":which}).count(), "records")
	while True:
		i = 0
		X_metadata = []
		X_audio = []
		X_video = []
		X_desc = []
		X_comments = [ [] for _ in range(numComments)]
		X_channel = []
		Y = []
		for record in collection.find({'which': which}):
			metadata = record['metadata']['metadata']['statistics']
			if len(record['metadata']['metadata']['comments']) < numComments:
				continue
			metadata_records = filled_metadata(metadata) + [record['metadata']['metadata']['categoryId']]
			metadata_records = np.array([int(x) for x in metadata_records])
			X_metadata.append(metadata_records)
			# Add metadata to training data point
			audio_data = record['avdata']['mean_audio']
			X_audio.append(audio_data)
			# Add audio features
			video_data = record['avdata']['mean_rgb']
			X_video.append(video_data)
			# Add video features
			description = d2vmodel.infer_vector(record['metadata']['metadata']['description'].split(' '))
			X_desc.append(description)
			# Add description features
			selectedComments = rankAndSelectComments(record['metadata']['metadata']['comments'], numComments)
			for i in range(numComments):
				X_comments[i].append(d2vmodel.infer_vector(selectedComments[i].split(' ')))
			# Add comment based features
			channelTitle = d2vmodel.infer_vector(record['metadata']['metadata']['channelTitle'].split(' '))
			X_channel.append(channelTitle)
			# Add channel title features
			Y.append(d2vmodel.infer_vector(record['metadata']['metadata']['title'].split(' ')))
			if len(Y) == batch_size:
				X = []
				if (use_metadata):
					X.append(np.array(X_metadata))
				if (use_audio):
					X.append(np.array(X_audio))
				if (use_video):
					X.append(np.array(X_video))
				if (use_desc):
					X.append(np.array(X_desc))
				if (use_comments):
					X += [ np.array(commentVec) for commentVec in X_comments ]
				if (use_channel):
					X.append(np.array(X_channel))
				Y = np.array(Y)
				# Split into train and validation if training
				if which == 1:
					splitPoint = int(validation_ratio * len(Y))
					X_train = [ x[splitPoint:] for x in X]
					X_val = [ x[:splitPoint] for x in X]
					yield (X_train, Y[splitPoint:]), (X_val, Y[:splitPoint])
				else:
					yield X, Y
				X_metadata, X_audio, X_video, X_desc, X_channel = [], [], [], [], []
				Y, X_comments = [], [ [] for _ in range(numComments) ]
		# Don't loop indefinitely if test data
		if which != 1:
			break


def getTrainValGens(sourceGen, train=True):
	if train:
		for tuple in sourceGen:
			yield tuple[0]
	else:
		for tuple in sourceGen:
			yield tuple[1]


if __name__ == "__main__":
	from pymongo import MongoClient
	client = MongoClient()
	db = client.youtube8m
	ds = db.iteration2
	d2v = doc2vec.Doc2Vec.load('doc2vec_100.model')
	gen = mongoDBgenerator(ds, d2v, 2, 1, 16) 
	x, y = gen.next()
	print x.shape, y.shape

