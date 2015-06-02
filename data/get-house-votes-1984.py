#! /usr/bin/env python
# coding: utf-8

import os
import urllib

def download(url, dst_path=None):
    """Download file from url to dst_path"""
    if dst_path is None:
        dst_path = url.split('/')[-1]
    elif os.path.isdir(dst_path):
        dst_path = os.path.join(dst_path, url.split('/')[-1])

    try:
        urllib.urlretrieve(url, dst_path)
        return dst_path
    except:
        if os.path.exists(dst_path):
            os.remove(dst_path)
        raise

download('https://raw.githubusercontent.com/j2kun/decision-trees/master/house-votes-1984.txt')