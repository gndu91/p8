from copy import deepcopy
from os import environ
from pathlib import Path

environ['CUDA_VISIBLE_DEVICES'] = '-1'

from requests import get, Response
from hashlib import sha256
from tqdm import tqdm
from zipfile import ZipFile
import pandas as pd
_cache_folder = Path('~/.cache/gn_p7').expanduser()
_cache_folder.mkdir(parents=True, exist_ok=True)

_ds_url = 'https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip'

random_state: int = 42


def download(url: str) -> Path:
    url_id: str = sha256(url.encode('utf-8')).hexdigest()
    local_path: Path = _cache_folder / url_id
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        tmp_path: Path = _cache_folder / (url_id + '.tmp')
        res: Response = get(url, stream=True)
        with tmp_path.open('wb') as f, tqdm(
                total=int(res.headers.get('content-length')),
                desc=f'Downloading {url}',
                unit_scale=True) as q:
            for chunk in res.iter_content(chunk_size=8192):
                q.update(len(chunk))
                f.write(chunk)
        tmp_path.replace(local_path)
    return local_path


def download_zip_archive(url: str) -> Path:
    """Download a zip archive, extract it then return the folder containing its content"""
    archive_path: Path = download(url)
    archive_folder: Path = Path(archive_path.as_posix() + '.dir')

    if not archive_folder.exists():
        print(f'Extracting archive {url}...', flush=True)
        archive_temp: Path = Path(archive_path.as_posix() + '.tmp')
        archive_temp.mkdir(parents=True, exist_ok=True)
        archive: ZipFile = ZipFile(archive_path)
        archive.extractall(path=archive_temp)
        archive_temp.replace(archive_folder)
        print(f'Extracting archive {url}...done', flush=True)

    return archive_folder


datasets: dict[str, pd.DataFrame] = {}


def get_dataset(name: str) -> pd.DataFrame:
    folder = download_zip_archive(_ds_url)
    if not name.endswith('.csv'):
        name = f'{name}.csv'
    try:
        return datasets[name]
    except KeyError:
        try:
            _df = pd.read_csv(folder / name)
        except FileNotFoundError:
            raise KeyError(name) from None
        else:
            datasets[name] = _df
            return _df.copy()
