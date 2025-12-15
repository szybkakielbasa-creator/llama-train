#!/usr/bin/env python3
# download_gguf.py - download GGUF file

import sys
import urllib.request
import ssl

# Disable SSL verification if needed
ssl._create_default_https_context = ssl._create_unverified_context

url = sys.argv[1]
file_path = sys.argv[2]

try:
    print(f"Downloading {url} to {file_path}")
    urllib.request.urlretrieve(url, file_path)
    print("Download successful")
except Exception as e:
    print(f"Download failed: {e}")
    sys.exit(1)