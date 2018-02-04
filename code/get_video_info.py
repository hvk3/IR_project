import httplib2
import os
import sys

from apiclient.discovery import build_from_document
from apiclient.errors import HttpError
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow


CLIENT_SECRETS_FILE = "client_secrets.json"

YOUTUBE_READ_WRITE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

MISSING_CLIENT_SECRETS_MESSAGE = """
WARNING: Please configure OAuth 2.0
To make this sample run you will need to populate the client_secrets.json file
found at:
   %s
with information from the APIs Console
https://console.developers.google.com
For more information about the client_secrets.json file format, please visit:
https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
""" % os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   CLIENT_SECRETS_FILE))

def get_authenticated_service(args):
  flow = flow_from_clientsecrets(CLIENT_SECRETS_FILE, scope=YOUTUBE_READ_WRITE_SSL_SCOPE,
    message=MISSING_CLIENT_SECRETS_MESSAGE)

  storage = Storage("%s-oauth2.json" % sys.argv[0])
  credentials = storage.get()

  if credentials is None or credentials.invalid:
    credentials = run_flow(flow, storage, args)

  with open("youtube-v3-api-captions.json", "r") as f:
    doc = f.read()
    return build_from_document(doc, http=credentials.authorize(httplib2.Http()))


def list_captions(youtube, video_id, reqd_language='en'):
  results = youtube.captions().list(
    part="snippet",
    videoId=video_id
  ).execute()

  for item in results["items"]:
    id = item["id"]
    name = item["snippet"]["name"]
    language = item["snippet"]["language"]
    if language == reqd_language:
        return id
  return None


def download_caption(youtube, caption_id, tfmt):
  subtitle = youtube.captions().download(
    id=caption_id,
    tfmt=tfmt
  ).execute()
  return  subtitle


def get_stats(youtube, video_id):
    response = youtube.videos().list(
      part='statistics, snippet',
      id=video_id
    ).execute()

    snippet = response['items'][0]['snippet']
    statistics = response['items'][0]['statistics']
    title = snippet.get('title', None)
    channelTitle = snippet.get('channelTitle', None)
    categoryId = snippet.get('categoryId', None)
    description = snippet.get('description', None)
    tags = snippet.get('tags', None)
    return title, channelTitle, categoryId, description, tags, statistics


def get_comment_threads(youtube, video_id):
  results = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    textFormat="plainText"
  ).execute()

  comments = []

  for item in results["items"]:
    comment = item["snippet"]["topLevelComment"]
    like_count = comment['snippet']['likeCount']
    author = comment["snippet"]["authorDisplayName"]
    text = comment["snippet"]["textDisplay"]
    comment_info = {
        "likes": like_count
        ,"author": author
        ,"comment": text
    }
    comments.append(comment_info)

  return comments


def get_video_metadata(youtube, videoid):
    comments = get_comment_threads(youtube, videoid)
    title, channelTitle, categoryId, description, tags, statistics = get_stats(youtube, videoid)
    metadata = {
        "comments": comments
        ,"title": title
        ,"channelTitle": channelTitle
        ,"categoryId": categoryId
        ,"description": description
        ,"tags": tags
        ,"statistics": statistics
    }
    return metadata


if __name__ == "__main__":
  pseudoargs = {"videoid": "elzrhKPrUh0", "language": "en"}
  youtube = get_authenticated_service(pseudoargs)
  metadata = get_video_metadata(youtube, pseudoargs['videoid'])
  print metadata
