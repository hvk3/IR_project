import pymongo
import json
import base64
import numpy as np
import os
import tensorflow as tf
from google.protobuf.json_format import MessageToJson

LOC = '/home/anshuman14021/IRP/data/yt8m_video_level'

client = pymongo.MongoClient() # defaults to port 27017
db = client.youtube8m
ds = db.iteration1


# dataKeys = db.iteration1.find({})

# for key in dataKeys:
#        print key
#       # print type(key)
# print " printed keys"

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

def generator(which_generator, batch_size):
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

        while True:
                i %= batch_size
                json_parsed_records = []
                for record in protobuf_records[i * batch_size : (i + 1) * batch_size]:
                        for record_ in tf.python_io.tf_record_iterator(record):
                                json_parsed_record = json.loads(MessageToJson(tf.train.Example.FromString(record_)))

                                videoId = base64.b64decode(extract_relevant_info(json_parsed_record)['video_id'])
                                if not ds.find({'_id': videoId}).count():
                                    #print " not in mongo db "
                                     pass
                                else:
                                    json_parsed_records.append(extract_relevant_info(json_parsed_record))
                i += 1
                if (len(json_parsed_records) == 0):
                        continue
                yield np.array(json_parsed_records)


if __name__ == "__main__":

        gen = generator(1, 1)
        arr = gen.next()

# print the number of documents in a collection
print db.iteration1.count()
