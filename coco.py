import os
from pathlib import Path
#  from pprint import pprint
from .utils import download_from_url
from .utils import extract_from_zipfile

COCO_ZIP_FILE_URLS = {
    #  '2014 Train images': 'http://images.cocodataset.org/zips/train2014.zip',
    '2014 Val images':   'http://images.cocodataset.org/zips/val2014.zip',
    #  '2014 Test images':  'http://images.cocodataset.org/zips/test2014.zip',
    '2014 Train/Val annotations':
        'http://images.cocodataset.org/annotations/'
        'annotations_trainval2014.zip',
    }

path_data_directory = Path.cwd()  # Path('/data/COCO/')


for coco_zip_file_url in COCO_ZIP_FILE_URLS.values():
    fname = os.path.basename(coco_zip_file_url)
    path_zip_file = path_data_directory.joinpath(fname)
    download_from_url(coco_zip_file_url, path_zip_file)
    extract_from_zipfile(path_zip_file)
