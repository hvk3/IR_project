import json
import numpy as np
import os
import base64
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from gensim.models import doc2vec


def extract_relevant_info(json_parsed_record, which_generator):
	video_id = base64.b64decode(json_parsed_record['features']['feature']['video_id']['bytesList']['value'][0])
	labels = json_parsed_record['features']['feature']['labels']['int64List']['value']
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
		print j, "out of", len(protobuf_records) // batch_size, "done"
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


def mongoDBgenerator(collection, d2vmodel, numComments, which, batch_size):
	while True:
		i = 0
		X = []
		Y = []
		for record in collection.find():
			if record['metadata']['which'] == which:
				record_X = []
				record_Y = []
				metadata = record['metadata']['metadata']['statistics']
				metadata_records = [ metadata['commentCount'], metadata['viewCount'], metadata['favoriteCount'], metadata['dislikeCount'], metadata['likeCount'], record['metadata']['metadata']['categoryId']]
				metadata_records = [int(x) for x in metadata_records]
				record_X = metadata_records
				# Add metadata to training data point
				audio_data = record['avdata']['mean_audio']
				record_X += audio_data
				# Add audio features
				video_data = record['avdata']['mean_rgb']
				record_X += video_data
				# Add video features
				description = d2vmodel.infer_vector(record['metadata']['metadata']['description'].split(' '))
				record_X += [description]
				# Add description features
				selectedComments = rankAndSelectComments(record['metadata']['metadata']['comments'], numComments)				
				vectorComments = [d2vmodel.infer_vector(x.split(' ')) for x in selectedComments]
				record_X += vectorComments
				# Add comment based features
				channelTitle = d2vmodel.infer_vector(record['metadata']['metadata']['channelTitle'].split(' '))
				record_X += [channelTitle]
				# Add channel title features
				record_Y = d2vmodel.infer_vector(record['metadata']['metadata']['title'].split(' '))
				X.append(record_X)
				Y.append(record_Y)
				if len(Y) == batch_size:
					yield np.array(X), np.array(Y)
					X = []
					Y = []


if __name__ == "__main__":
	from pymongo import MongoClient
	client = MongoClient()
	db = client.youtube8m
	ds = db.iteration2
	d2v = doc2vec.Doc2Vec.load('doc2vec_100.model')
	gen = mongoDBgenerator(ds, d2v, 2, 1, 16) 
	x, y = gen.next()
	print x.shape, y.shape
	print x
