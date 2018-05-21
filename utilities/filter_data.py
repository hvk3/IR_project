from pymongo import MongoClient
from langdetect import detect
from tqdm import tqdm

client = MongoClient()
db = client.youtube8m
ds_1 = db.iteration3
ds_2 = db.iteration4

ds_2.remove()
print("Before:", ds_1.find().count())
for record in tqdm(ds_1.find()):
	title = record['metadata']['title']
	description = record['metadata']['description']
#	if len(description) > 0 and len(title) > 0:
#		ds_2.insert_one(record)
	try:
		if detect(description) == 'en': #3: title, #4: description
			ds_2.insert_one(record)
	except:
		continue

print("After:", ds_2.find().count())
