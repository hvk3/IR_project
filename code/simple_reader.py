import json
import numpy as np
import os
import tensorflow as tf
from google.protobuf.json_format import MessageToJson

def generator(protobuf_records, which_generator, batch_size):
	loc = os.path.join('home', 'anshuman14021', 'IRP', 'data', 'yt8m_video_level')
	protobuf_records = filter(lambda x: 'tfrecord' in x, os.listdir(loc))
	if (which_generator == 1):
		protobuf_records = filter(lambda x: 'train' in x, protobuf_records)
	elif (which_generator == 2):
		protobuf_records = filter(lambda x: 'test' in x, protobuf_records)
	elif (which_generator == 3):
		protobuf_records = filter(lambda x: 'validate' in x, protobuf_records)
	protobuf_records = map(lambda x: os.path.join(loc, x), protobuf_records)
	i = 0
	while True:
		i %= batch_size
		json_parsed_records = []
		for record in protobuf_records[i * batch_size : (i + 1) * batch_size]:
			for record_ in tf.python_io.tf_record_iterator(record):
				json_parsed_record = json.loads(MessageToJson(tf.train.Example.FromString(record_)))
				json_parsed_records.append(json_parsed_record)
		i += 1
		if (len(json_parsed_records) == 0):
			continue
		yield np.array(json_parsed_records)
		# access element x using json_parsed_record['features']['feature'][x]
