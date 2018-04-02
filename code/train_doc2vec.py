import argparse
from gensim.models import doc2vec
from pymongo import MongoClient

parser = argparse.ArgumentParser()
parser.add_argument(
    '--embedding_size',
    type=int,
    default=100
)


class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            yield doc2vec.LabeledSentence(
                words=line.split(),
                labels=['SENT_%s' % uid]
            )


def get_batches_from_collection(collection_, batch_size=16):
    items = collection_.find()
    total_items = items.count()
    i = 0
    while (True):
        i %= (total_items // batch_size)
        items_batch = items[i * batch_size:(i + 1) * batch_size]
        i += 1
        print(i)
        yield items_batch


if __name__ == '__main__':
    args = parser.parse_args()
    embedding_size = args.embedding_size
    client = MongoClient()
    db = client['youtube8m']
    collection_ = db['iteration2']
    items = collection_.find()
    total_items = items.count()
    print(total_items, "in mongoDB for consideration")
    sentences = map(
        lambda x: x['metadata']['metadata'].get('description', ''),
        items
    )
    print("Done one pass of mapping sentences")
    sentences_ = map(
        lambda x: doc2vec.TaggedDocument(words=x[1].split(), tags=[x[0]]),
        enumerate(sentences)
    )
    print("Done another pass of mapping sentences")
    model = doc2vec.Doc2Vec(alpha=0.1, min_alpha=0.01, size=embedding_size)
    model.build_vocab(documents=sentences_)
    model.train(sentences_, total_examples=model.corpus_count, epochs=100)
    model.save('doc2vec_{}.model'.format(embedding_size))
    print('Model Saved')

