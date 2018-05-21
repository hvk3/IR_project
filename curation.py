from pymongo import MongoClient
import numpy as np

client = MongoClient()
ds = client.youtube8m
db = ds.iteration1
a = []

#print db.find_one({'_id': '0l5uZ-JDYDA'})
#exit()

for record in db.find():
	if len(record['metadata']['comments']) != 0:
		a.append(record['_id'])

np.random.shuffle(a)
for i in a:
	print i
	raw_input()
