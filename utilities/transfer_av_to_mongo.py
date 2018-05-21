import pymongo
import json
import base64
import numpy as np
import os
import tensorflow as tf

from google.protobuf.json_format import MessageToJson
from tqdm import tqdm

LOC = '/home/anshuman14021/IR_project/data/yt8m_video_level'
client = pymongo.MongoClient() # defaults to port 27017
db = client.youtube8m
ds = db.av_features
ref_ds = db.iteration3

def extract_relevant_info(json_parsed_record):
        video_id = json_parsed_record['features']['feature']['video_id']['bytesList']['value'][0]
        labels = json_parsed_record['features']['feature']['labels']['int64List']['value']
        mean_audio = json_parsed_record['features']['feature']['mean_audio']['floatList']['value']
        mean_rgb = json_parsed_record['features']['feature']['mean_rgb']['floatList']['value']
        parsed_record = {
                "video_id": video_id
                ,"labels": labels
                ,"mean_audio": mean_audio
                ,"mean_rgb": mean_rgb
        }
        return parsed_record


def generator(which_generator):
        global LOC

        protobuf_records = filter(lambda x: 'tfrecord' in x, os.listdir(LOC))
        if (which_generator == 1):
                protobuf_records = filter(lambda x: 'train' in x, protobuf_records)
        elif (which_generator == 2):
                protobuf_records = filter(lambda x: 'test' in x, protobuf_records)
        elif (which_generator == 3):
                protobuf_records = filter(lambda x: 'validate' in x, protobuf_records)
        protobuf_records = map(lambda x: os.path.join(LOC, x), protobuf_records)

        for record in tqdm(protobuf_records):
            for record_ in tf.python_io.tf_record_iterator(record):
                json_parsed_record = json.loads(MessageToJson(tf.train.Example.FromString(record_)))
                json_stuff = extract_relevant_info(json_parsed_record)
                videoId = base64.b64decode(extract_relevant_info(json_parsed_record)['video_id'])
                if ref_ds.find_one({'_id': videoId}):
                    ds.insert_one({
                        '_id': videoId,
                        'mean_audio': json_stuff['mean_audio'],
                        'mean_rgb': json_stuff['mean_rgb']})
                #print(ds.find_one({'_id': videoId})['mean_audio'])
                #print(videoId)

if __name__ == "__main__":
    ds.remove()
    generator(1)
