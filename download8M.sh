#!/bin/bash

mkdir -p data/yt8m_video_level; cd data/yt8m_video_level


curl data.yt8m.org/download.py | partition=1/video_level/train mirror=asia python
curl data.yt8m.org/download.py | partition=1/video_level/validate mirror=asia python
curl data.yt8m.org/download.py | partition=1/video_level/test mirror=asia python
