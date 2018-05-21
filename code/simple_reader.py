import json
import numpy as np
import os
import base64
import pickle
import string
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from google.protobuf.json_format import MessageToJson
from gensim.models import doc2vec

# For preparing text for LSTM model, use this
# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/


def get_titles(collection):
    # import pdb
    if (not os.path.exists('titles.npy')):
        # pdb.set_trace()
        titles = []
        for record in collection.find({}):
            title = record['metadata']['title']
            title = ''.join([i if ord(i) < 128 else '' for i in title])
            title = title.lower().strip()
            title = filter(lambda x: x not in string.punctuation, title)
            if (len(title.split(' ')) > 10):
                continue
            titles.append(title)
        np.save('titles.npy', titles)
    else:
        titles = np.load('titles.npy')
    return titles


def prune(titles, tokenizer):
    pruned_titles = []
    for title in titles:
        flag = False
        for token in title.split(' '):
            if (tokenizer.word_counts.get(token, 0) <= 1):
                flag = True
                break
        if (not flag):
            pruned_titles.append(title)
    print len(titles), len(pruned_titles)
    np.save('titles.npy', pruned_titles)


def pickle_tokenizer(collection):
    titles = get_titles(collection)
    if (not os.path.exists('tokenizer.pickle')):
        tokenizer = Tokenizer(num_words=50000)
        tokenizer.fit_on_texts(titles)
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        prune(titles, tokenizer)
    else:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    return tokenizer


def extract_relevant_info(json_parsed_record, which_generator):
    features = json_parsed_record['features']['feature']
    video_id = base64.b64decode(features['video_id']['bytesList']['value'][0])
    # video_id = features['video_id']['bytesList']['value'][0]
    try:
        labels = features['labels']['int64List']['value']
    except KeyError:
        labels = []
    mean_audio = features['mean_audio']['floatList']['value']
    mean_rgb = features['mean_rgb']['floatList']['value']
    parsed_record = {
        "video_id": video_id,
        "labels": labels,
        "mean_audio": mean_audio,
        "mean_rgb": mean_rgb,
        "which_generator": which_generator
    }
    return parsed_record


def generator(location, which_generator, batch_size):
    protobuf_records = filter(lambda x: 'tfrecord' in x, os.listdir(location))
    if (which_generator == 1):
        protobuf_records = filter(lambda x: 'train' in x, protobuf_records)
    elif (which_generator == 2):
        protobuf_records = filter(lambda x: 'test' in x, protobuf_records)
    elif (which_generator == 3):
        protobuf_records = filter(lambda x: 'validate' in x, protobuf_records)
    protobuf_records = map(
        lambda x: os.path.join(location, x),
        protobuf_records
    )
    i = 0

    for j in range(len(protobuf_records) // batch_size):
        print(j, "out of", len(protobuf_records) // batch_size, "done")
        i %= batch_size
        json_parsed_records = []
        for record in protobuf_records[i * batch_size:(i + 1) * batch_size]:
            for record_ in tf.python_io.tf_record_iterator(record):
                json_parsed_record = json.loads(
                    MessageToJson(tf.train.Example.FromString(record_))
                )
                json_parsed_records.append(
                    extract_relevant_info(json_parsed_record, which_generator)
                )
        i += 1
        if (len(json_parsed_records) == 0):
            continue
        yield np.array(json_parsed_records)


def rankAndSelectComments(comments, select):
    likes = []
    for comment in comments:
        likes.append(int(comment['likes']))
    sortedByLikes = np.argsort(likes)[::-1]
    selectedComments = []
    for i in range(len(sortedByLikes)):
        selectedComments.append(comments[sortedByLikes[i]]['comment'])
    return selectedComments[:select]


def filled_metadata(metadata):
    commentCount = metadata.get('commentCount', 0)
    viewCount = metadata.get('viewCount', 0)
    favoriteCount = metadata.get('favoriteCount', 0)
    dislikeCount = metadata.get('dislikeCount', 0)
    likeCount = metadata.get('likeCount', 0)
    return [commentCount, viewCount, favoriteCount, dislikeCount, likeCount]


def mongoDBgenerator(
    collection,
    d2vmodel,
    batch_size,
    validation_ratio,
    use_titles,
    maxlen,
    is_train_data=True
):
    # print "Found {} records".format(collection.find({"which":which}).count())
    if (use_titles):
        tokenizer = pickle_tokenizer(collection)
        vocab_size = tokenizer.num_words
        pruned_titles = set(np.load('titles.npy').tolist())
    while True:
        X_audio, X_video, X_desc =\
            [[] for _ in xrange(3)]
        Y = []
        for record in collection.find():
            metadata = record['metadata']
            if (not use_titles):
                Y.append(d2vmodel.infer_vector(metadata['title'].split(' ')))
            else:
                title = metadata['title']
                title = ''.join([i if ord(i) < 128 else '' for i in title])
                title = title.lower().strip()
                title = filter(lambda x: x not in string.punctuation, title)
                if (title not in pruned_titles):
                    continue
                Y_ = pad_sequences(
                    tokenizer.texts_to_sequences([title]),
                    maxlen=maxlen,
                    padding='post',
                    truncating='post'
                )
                Y.append(np.array(map(
                    lambda x: map(
                        lambda y: to_categorical(y, vocab_size),
                        x
                    ),
                    Y_
                )))
                del Y_
            # Add audio features
            X_audio.append(record['mean_audio'])
            # Add video features
            X_video.append(record['mean_rgb'])
            # Add description based features
            X_desc.append(
                d2vmodel.infer_vector(metadata['description'].split(' '))
            )
            #print(d2vmodel.infer_vector(metadata['description'].split(' '))).shape
            if len(X_desc) == batch_size:
                X = [
                    np.array(X_audio),
                    np.array(X_video),
                    np.array(X_desc)
                ]
                # Return all data from generator, handle in model
                Y = np.array(Y)
                if use_titles:
                    Y = np.squeeze(Y, axis=1)
                # Split into train and validation if training
                if is_train_data == True:
                    splitPoint = int(validation_ratio * len(Y))
                    X_train = [x[splitPoint:] for x in X]
                    X_val = [x[:splitPoint] for x in X]
                    yield (X_train, Y[splitPoint:]), (X_val, Y[:splitPoint])
                else:
                    yield (X, Y), (record['_id'], metadata['title'])
                X_audio, X_video, X_desc =\
                    [[] for _ in xrange(3)]
                Y = []
        # Don't loop indefinitely if test data
        if which != 1:
            break


def getTrainValGens(sourceGen, train=True):
    if train:
        for tuple in sourceGen:
            yield tuple[0]
    else:
        for tuple in sourceGen:
            yield tuple[1]

