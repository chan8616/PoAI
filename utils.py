import os
import requests
import zipfile
from tqdm import tqdm


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """

    file_size = int(requests.head(url).headers['Content-Length'])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


def extract_from_zipfile(path_zip_file):
    assert zipfile.is_zipfile(path_zip_file)
    first_byte = 0
    with zipfile.ZipFile(path_zip_file, 'r') as zf:
        zf_size = sum(f.file_size for f in zf.infolist())
        from pathlib import Path
        with tqdm(
                total=zf_size, initial=first_byte,
                unit='B', unit_scale=True, desc=path_zip_file.name) as pbar:

            for f in zf.infolist():
                zf.extract(f)
                pbar.update(f.file_size)
            #  with(open(dst, 'ab')) as f:
            #      for chunk in req.iter_content(chunk_size=1024):
            #          if chunk:
            #              f.write(chunk)
            #              pbar.update(1024)
            #      print(f)
            #      print(f.file_size)
            #      break
        #  zf.extractall()

    #  header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    #  pbar = tqdm(
    #      total=file_size, initial=first_byte,
    #      unit='B', unit_scale=True, desc=url.split('/')[-1])
    #  req = requests.get(url, headers=header, stream=True)
    #  with(open(dst, 'ab')) as f:
    #      for chunk in req.iter_content(chunk_size=1024):
    #          if chunk:
    #              f.write(chunk)
    #              pbar.update(1024)
    #  pbar.close()
    #  return file_size
