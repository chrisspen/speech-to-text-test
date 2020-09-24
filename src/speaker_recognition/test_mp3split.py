#!/usr/bin/env python
"""
sudo apt install mp3splt
"""
import os
os.system("cd data2; time mp3splt sample1.mp3 -s -p min=0.5,threshold=-16")
