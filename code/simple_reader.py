import json
import numpy as np
import os
import base64
import tensorflow as tf
from google.protobuf.json_format import MessageToJson


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
		print j, "out of", len(protobuf_records) // batch_size, "read"
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


if __name__ == "__main__":
	location = '/home/anshumans/Desktop/Studies/IR/IR_project/data/dummydata'
	gen = generator(location, 1, 1)
	arr = gen.next()
	print arr[0]
