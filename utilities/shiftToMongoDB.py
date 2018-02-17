import pymongo
import json
import base64
import numpy as np
import os
import tensorflow as tf
from google.protobuf.json_format import MessageToJson

LOC = '/home/anshuman14021/IRP/data/yt8m_video_level'

client = pymongo.MongoClient()
db = client.youtube8m
ds = db.iteration1
ds2 = db.iteration2


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


def cache_everything(which_generator):
        global LOC
        protobuf_records = filter(lambda x: 'tfrecord' in x, os.listdir(LOC))
        if (which_generator == 1):
                protobuf_records = filter(lambda x: 'train' in x, protobuf_records)
        elif (which_generator == 2):
                protobuf_records = filter(lambda x: 'test' in x, protobuf_records)
        elif (which_generator == 3):
                protobuf_records = filter(lambda x: 'validate' in x, protobuf_records)
        protobuf_records = map(lambda x: os.path.join(LOC, x), protobuf_records)
        i = 0
        for record in protobuf_records:
            for record_ in tf.python_io.tf_record_iterator(record):
                json_parsed_record = json.loads(MessageToJson(tf.train.Example.FromString(record_)))
                videoInfo = extract_relevant_info(json_parsed_record, which_generator)
                if ds.find({'_id': videoInfo['video_id']}).count():
                    ds2.insert_one({
                        "metadata": ds.find_one({'_id': videoInfo['video_id']})
                        ,"avdata": videoInfo
                    })
                    i += 1
                    if i == ds.find().count():
                        break
		    print i
                else:
                    continue


if __name__ == "__main__":
    gen = cache_everything(1)
