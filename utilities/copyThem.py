from pymongo import MongoClient

client = MongoClient()
db = client.youtube8m
av = db.av_features
train = db.train_ids
test = db.test_ids

TRAIN = db.train_split
TEST = db.test_split

TRAIN.remove()
TEST.remove()

for record in train.find():
	id = record['_id']
	rec = av.find_one({'_id': id})
	record['mean_audio'] = rec['mean_audio']
	record['mean_rgb'] = rec['mean_rgb']
	TRAIN.insert_one(record)


for record in test.find():
        id = record['_id']
        rec = av.find_one({'_id': id})
        record['mean_audio'] = rec['mean_audio']
        record['mean_rgb'] = rec['mean_rgb']
        TEST.insert_one(record)
