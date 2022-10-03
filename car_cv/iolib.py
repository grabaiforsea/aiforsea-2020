import tarfile
from sys import stdout

import requests


def stream_download(url: str, output_path: str):
    """
    Downloads a file in a streaming fashion (writing data to disk as it is received and then discarding it from
    memory).

    Args:
        url: The URL at which to find the file to be downloaded.
        output_path: The path where the downloaded file will be placed.
    """
    with requests.get(url, stream=True) as req:
        if req.status_code == 200:
            total_size = int(req.headers.get('content-length', None))
            downloaded_size = 0
            with open(output_path, 'wb') as f:
                for chunk in req.iter_content(chunk_size=4096):
                    downloaded_size += len(chunk)
                    if total_size:
                        output_string = (f'\rDownload progress (bytes): {downloaded_size:,}/{total_size:,} '
                                         f'({downloaded_size / total_size:.1%})')

                    else:
                        output_string = f'\rDownload progress (bytes): {downloaded_size:,}'

                    stdout.write(output_string)
                    stdout.flush()
                    f.write(chunk)

                stdout.write('\n')

        else:
            raise RuntimeError(f'Download failed; got HTTP code {req.status}')


def extract_tgz(archive_path: str, output_path: str):
    """
    Extracts files from a gzip-compressed tarball.

    Args:
        archive_path: The path to the archive.
        output_path: The path where the extracted files will be placed.
    """
    with tarfile.open(archive_path, 'r:gz') as f:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, output_path)
