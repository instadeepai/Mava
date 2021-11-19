#!/bin/bash
# Optional Install `xvfb` to run a headless screen to record agents.
apt-get update -y && apt-get install -y xvfb && apt-get install -y python-opengl && apt-get install -y ffmpeg
pip install pyglet
