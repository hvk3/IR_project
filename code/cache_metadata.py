from pymongo import MongoClient

import get_video_info
import simple_reader

def populate_data(location, which_generator):
    # Set up mongoDB
    client = MongoClient()
    db = client['youtube8m']
    ds = db['iteration1']

    # Read data files from generator
    gen = simple_reader.generator(location, which_generator, 16)

    i = 0
    # Populate mongoDB database
    while True:
        print "Processed", i
        try:
            vid_data = gen.next()
            for vid in vid_data:
                try:
                        videoid = vid['video_id']
                        which = vid['which_generator']
                        pseudoargs = {"videoid": videoid, "language": "en"}
                        # Query Google if it doesn't exist in cached DB
                        if not ds.find({'_id': videoid}).count():
                                youtube = get_video_info.get_authenticated_service(pseudoargs)
                                metadata = get_video_info.get_video_metadata(youtube, pseudoargs['videoid'])
                                ds.insert_one({
                                        "_id": videoid
                                        ,"metadata": metadata
                                        ,"which": which})
                                i += 1
                                print videoid
                        else:
                                continue
                except Exception, e:
			print e
                        continue
        except Exception, e:
	    print e
            break

if __name__ == "__main__":
    import sys
    populate_data(sys.argv[1], 2)
