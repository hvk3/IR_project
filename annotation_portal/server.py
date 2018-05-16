from flask import Flask, render_template, request
from pymongo import MongoClient

import json
import numpy as np
import os

app = Flask(__name__, template_folder='.')
client = MongoClient()


@app.route('/', methods=['GET', 'POST'])
def render_video():
    try:
        idx = int(request.args.values()[0])
    except IndexError:
        import ast
        idx = int(ast.literal_eval(request.form.keys()[0])['idx'])
    video_id_mappings = client['videos'].video_ids
    video = video_id_mappings.find({})[idx]
    filename = os.path.join('static', video['id'])
    options = video['options']
    np.save('idx', idx)
    html = render_template(
        'index.html',
        video_name=filename,
        options=options
    )
    return html


@app.route('/update', methods=['POST'])
def update():
    form = request.form
    jsonForm = json.loads(list(form)[0])
    dict_ = {}
    dict_['a'] = int(jsonForm['a'])
    dict_['b'] = int(jsonForm['b'])
    dict_['c'] = int(jsonForm['c'])
    dict_['d'] = int(jsonForm['d'])
    reranked_options = map(
        lambda x: x[0],
        sorted(dict_.iteritems(), key=lambda y: y[1], reverse=True)
    )
    video_id_mappings = client['videos'].video_ids
    video = jsonForm['video'].split('/')[-1]
    video_id_mappings.update_one(
        {'id': video},
        {'$set': {'reranked_options': reranked_options}},
        upsert=True
    )
    idx = np.load('idx.npy').item()
    return json.dumps({'idx': idx + 1})


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
