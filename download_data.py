"""Downloads a public dataset into ``data/<dataset>/``.

Usage: python download_data.py DATASET_NAME

Supported datasets: ml-100k, ml-1m, ml-10m, tripadvisor-<city> (gijon, barcelona, madrid, newyork, paris,
london), douban-monti and netflix-prize. GDSC1, CTRPv2, IMF DOTS and Kiva have to be obtained from their
original sources and preprocessed (see ``eauc/datasets.py`` for the expected files).
"""

import io
import os
import shutil
import sys
import tarfile
import zipfile

import requests

DATA_DIR = "data"
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/{name}.zip"
TRIPADVISOR_URL = "https://zenodo.org/record/5644892/files/{city}.zip?download=1"
DOUBAN_URL = "https://github.com/fmonti/mgcnn/blob/master/Data/douban/training_test_dataset.mat?raw=true"
NETFLIX_URL = "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz"


def download_and_extract(url: str, dataset_name: str) -> None:
    """Download ``url`` and extract it (zip or tar) under ``data/``; plain files (Douban) are saved as is."""
    print(f"Downloading and extracting {dataset_name} dataset...")
    content = requests.get(url, timeout=600).content

    if zipfile.is_zipfile(io.BytesIO(content)):
        zipfile.ZipFile(io.BytesIO(content)).extractall(DATA_DIR)
    elif tarfile.is_tarfile(io.BytesIO(content)):
        tarfile.open(fileobj=io.BytesIO(content)).extractall(DATA_DIR)
    else:
        os.makedirs(os.path.join(DATA_DIR, dataset_name), exist_ok=True)
        with open(os.path.join(DATA_DIR, dataset_name, "training_test_dataset.mat"), "wb") as f:
            f.write(content)


def create_netflix_ratings_csv() -> None:
    """Merge the per-movie files of the Netflix Prize training set into ``data/netflix-prize/ratings.csv``.

    The CSV has columns ``user_id, movie_id, rating``. The extracted dataset must already be in
    ``data/netflix-prize/``.
    """
    folder = os.path.join(DATA_DIR, "netflix-prize")
    archive = os.path.join(folder, "training_set.tar")
    print("Extracting training set from tar...")
    with tarfile.open(archive) as tar:
        tar.extractall(folder)
    os.remove(archive)

    training_set = os.path.join(folder, "training_set")
    with open(os.path.join(folder, "ratings.csv"), "w") as ratings:
        ratings.write("user_id,movie_id,rating\n")
        for filename in os.listdir(training_set):
            print(f"Merging training set files... {filename}", end="\r")
            # Each file starts with "<movie_id>:" followed by "<user_id>,<rating>,<date>" lines
            with open(os.path.join(training_set, filename)) as movie_file:
                movie_id = movie_file.readline().split(":")[0]
                for line in movie_file:
                    user_id, rating, _ = line.split(",")
                    ratings.write(f"{user_id},{movie_id},{rating}\n")

    shutil.rmtree(training_set)


def download_data(dataset_name: str) -> bool:
    """Download ``dataset_name``. Returns ``False`` if the dataset is not supported."""
    if dataset_name in ("ml-100k", "ml-1m"):
        download_and_extract(MOVIELENS_URL.format(name=dataset_name), dataset_name)
    elif dataset_name == "ml-10m":
        download_and_extract(MOVIELENS_URL.format(name=dataset_name), dataset_name)
        os.rename(os.path.join(DATA_DIR, "ml-10M100K"), os.path.join(DATA_DIR, "ml-10m"))
    elif dataset_name.startswith("tripadvisor-"):
        city = dataset_name.split("-")[1]
        download_and_extract(TRIPADVISOR_URL.format(city=city), dataset_name)
        os.rename(os.path.join(DATA_DIR, city), os.path.join(DATA_DIR, dataset_name))
    elif dataset_name == "douban-monti":
        download_and_extract(DOUBAN_URL, dataset_name)
    elif dataset_name == "netflix-prize":
        download_and_extract(NETFLIX_URL, dataset_name)
        os.rename(os.path.join(DATA_DIR, "download"), os.path.join(DATA_DIR, "netflix-prize"))
        create_netflix_ratings_csv()
    else:
        return False

    print(f"Dataset {dataset_name} downloaded and extracted successfully.")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    if not download_data(sys.argv[1]):
        sys.exit(f"Dataset {sys.argv[1]!r} is not supported.\n\n{__doc__}")
