from flask import Flask, render_template, request
from pymongo import MongoClient

import json
import numpy as np
import os
import string

app = Flask(__name__, template_folder='.')
client = MongoClient()


@app.route('/', methods=['GET', 'POST'])
def render_video():
    try:
        idx = int(request.args.values()[0])
    except IndexError:
        try:
            import ast
            idx = int(ast.literal_eval(request.form.keys()[0])['idx'])
        except IndexError:
            idx = 0
    video_id_mappings = client['videos_2'].video_ids
    videos = list(video_id_mappings.find())
    if (idx >= len(videos)):
        exit()
    video = videos[idx]
    filename = os.path.join('static', video['id'])
    titles = video['shuffled_titles']
    options = [string.ascii_lowercase[i] for i in xrange(len(titles))]
    np.save('idx_2', idx)
    html = render_template(
        'index_2.html',
        video_name=filename,
        options=options,
        titles=titles
    )
    return html


@app.route('/update', methods=['POST'])
def update():
    video_id_mappings = client['videos_2'].video_ids
    form = request.form
    jsonForm = json.loads(list(form)[0])
    reranked_options = [
        int(jsonForm['a']) - 1,
        int(jsonForm['b']) - 1,
        int(jsonForm['c']) - 1,
        int(jsonForm['d']) - 1
    ]
    # Reranked options are ranking given to shuffled titles
    import pdb
    #pdb.set_trace()
    print reranked_options
    video = jsonForm['video'].split('/')[-1]
    reranked_titles = []
    shuffled_titles = video_id_mappings.find_one(
        {'id': video})['shuffled_titles']
    shuffled_titles_with_rank = sorted(
        zip(shuffled_titles, reranked_options),
        key=lambda x: x[1]
    )
    for i in xrange(len(shuffled_titles_with_rank)):
        reranked_titles.append(shuffled_titles_with_rank[i][0])
    # reranked_titles = np.array(
    #     video_id_mappings.find_one({'id': video})['shuffled_titles']
    # )[reranked_options].tolist()
    video_id_mappings.update_one(
        {'id': video},
        {
            '$set':
                {'reranked_titles': reranked_titles, 'order': reranked_options}
        },
        upsert=True
    )
    idx = np.load('idx_2.npy').item()
    return json.dumps({'idx': idx + 1})


@app.route('/ip', methods=['GET'])
def name():
    return request.environ.get('HTTP_X_REAL_IP', request.remote_addr)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, host='0.0.0.0',port=7569)

