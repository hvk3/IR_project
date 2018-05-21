from pymongo import MongoClient
from flask import Flask, request, make_response, current_app
from gensim.models import doc2vec
from keras.models import load_model
from scipy import spatial

import json
import numpy as np

# Global variables
d2v = None
model = None
app = Flask(__name__)


@app.route("/get_similarity",  methods=['GET', 'POST'])
def analyze_image():
	global d2v, model
	video_id = request.args['video_id']
	suggested_title = request.args['suggested_title']
	client = MongoClient()
	db = client.youtube8m
	ds_1 = db.train_split
	ds_2 = db.test_split
	entry = ds_1.find_one({'_id': video_id})
	if not entry:
		entry = ds_2.find_one({'_id': video_id})
	if entry:
		a_v, v_v, d_v = entry['mean_audio'], entry['mean_rgb'], d2v.infer_vector(entry['metadata']['description'].split(' '))
		video_embedding = model.predict([np.array([a_v, a_v]),
			np.array([v_v, v_v]),
			np.array([d_v, d_v])
		])[0]
		title_embedding = d2v.infer_vector(suggested_title.split(' '))
		similarity = 1 - (0.5 * spatial.distance.cosine(video_embedding, title_embedding))
		result = {'score' : similarity}
		cache_it = json.dumps(result.copy())
		return cache_it
	else:
		#return json.dumps({"score": 0.8512})
		return json.dumps({"error": 102})


if __name__ == "__main__":
	import sys
	d2v = doc2vec.Doc2Vec.load('enwiki_dbow/doc2vec.bin')
	model = load_model(sys.argv[1])
	model.predict([np.zeros((1, 128)), np.zeros((1, 1024)), np.zeros((1, 300))])
	app.debug = False
	app.run(host = '0.0.0.0', port=1025)

