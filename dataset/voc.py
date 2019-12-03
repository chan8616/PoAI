import os
from pathlib import Path
#  from pprint import pprint
from .utils import download_from_url
from .utils import extract_from_zipfile

VOC_ZIP_FILE_URLS = {
    '2012 TrainVal images': 'http://host.robots.ox.ac.uk/pascal/VOC/'
                            'voc2012/VOCtrainval_11-May-2012.tar',
    #  '2014 Val images':   'http://images.cocodataset.org/zips/val2014.zip',
    #  '2014 Test images':  'http://images.cocodataset.org/zips/test2014.zip',
    #  '2014 Train/Val annotations':
    #      'http://images.cocodataset.org/annotations/'
    #      'annotations_trainval2014.zip',
    }

path_data_directory = Path.cwd()  # Path('/data/voc/')


for voc_zip_file_url in VOC_ZIP_FILE_URLS.values():
    fname = os.path.basename(voc_zip_file_url)
    path_zip_file = path_data_directory.joinpath(fname)
    download_from_url(voc_zip_file_url, path_zip_file)
    extract_from_zipfile(path_zip_file)
