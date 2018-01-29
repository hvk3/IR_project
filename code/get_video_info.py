import httplib2
import os
import sys

from apiclient.discovery import build_from_document
from apiclient.errors import HttpError
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow


CLIENT_SECRETS_FILE = "client_secrets.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
YOUTUBE_READ_WRITE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# This variable defines a message to display if the CLIENT_SECRETS_FILE is
# missing.
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

# Authorize the request and store authorization credentials.
def get_authenticated_service(args):
  flow = flow_from_clientsecrets(CLIENT_SECRETS_FILE, scope=YOUTUBE_READ_WRITE_SSL_SCOPE,
    message=MISSING_CLIENT_SECRETS_MESSAGE)

  storage = Storage("%s-oauth2.json" % sys.argv[0])
  credentials = storage.get()

  if credentials is None or credentials.invalid:
    credentials = run_flow(flow, storage, args)

  # Trusted testers can download this discovery document from the developers page
  # and it should be in the same directory with the code.
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
    title = snippet['title']
    channelTitle = snippet['channelTitle']
    categoryId = snippet['categoryId']
    description = snippet['description']
    tags = snippet['tags']
    return title, channelTitle, categoryId, description, tags, statistics


def get_comment_threads(youtube, video_id):
  results = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    textFormat="plainText"
  ).execute()

  for item in results["items"]:
    comment = item["snippet"]["topLevelComment"]
    author = comment["snippet"]["authorDisplayName"]
    text = comment["snippet"]["textDisplay"]
    print "Comment by %s: %s" % (author, text)

  return results["items"]

# Call the API's comments.list method to list the existing comment replies.
def get_comments(youtube, parent_id):
  results = youtube.comments().list(
    part="snippet",
    parentId=parent_id,
    textFormat="plainText"
  ).execute()

  for item in results["items"]:
    author = item["snippet"]["authorDisplayName"]
    text = item["snippet"]["textDisplay"]
    print "Comment by %s: %s" % (author, text)

  return results["items"]


if __name__ == "__main__":
  # The "videoid" option specifies the YouTube video ID that uniquely
  # identifies the video for which the caption track will be uploaded.
  argparser.add_argument("--videoid",
    help="Required; ID for video for which the caption track will be uploaded.")
  # The "name" option specifies the name of the caption trackto be used.
  argparser.add_argument("--name", help="Caption track name", default="YouTube for Developers")
  # The "language" option specifies the language of the caption track to be uploaded.
  argparser.add_argument("--language", help="Caption track language", default="en")
  args = argparser.parse_args()

  youtube = get_authenticated_service(args)

  # Comments
  video_comment_threads = get_comment_threads(youtube, args.videoid)
  parent_id = video_comment_threads[0]["id"]
  video_comments = get_comments(youtube, parent_id)

  # Metadata
  title, channelTitle, categoryId, description, tags, statistics = \
    get_stats(youtube, args.videoid)

  # Almost always unavailable:
  try:
      captionid = list_captions(youtube, args.videoid, args.language)
      if captionid:
          download_caption(youtube, captionid, 'srt')
  except HttpError, e:
    print "An HTTP error %d occurred:\n%s" % (e.resp.status, e.content)
