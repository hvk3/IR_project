from pymongo import MongoClient
import numpy as np
import shutil


if __name__ == '__main__':
    client = MongoClient()
    videoId_titles = np.genfromtxt(
        'videoId_title.txt',
        dtype=None,
        delimiter='\t'
    )
    import pdb
    # pdb.set_trace()
    video_id_mappings = client['videos_2'].video_ids
    iteration1 = client['youtube8m'].iteration1
    for i, videoId_title in enumerate(videoId_titles):
        videoId = videoId_title[0]
        titles = [iteration1.find_one(
            {'_id': videoId.encode('ascii')}
        )['metadata']['title']] + videoId_title[1:].tolist()
        shuffled_titles = np.copy(titles)
        np.random.shuffle(shuffled_titles)
        json_obj = {
            'id': '{}.mp4'.format(i),
            'original_titles': titles,
            'shuffled_titles': shuffled_titles.tolist()
        }
        video_id_mappings.insert_one(json_obj)
        shutil.copy2(
            'anno/{}'.format(videoId),
            'annotation_portal/static/{}.mp4'.format(i)
        )

