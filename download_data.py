# Downloads the requested dataset DATASET_NAME and saves it in the
# data/DATASET_NAME folder.

# Usage: python download_data.py DATASET_NAME (e.g. python download_data.py ml-100k)

# Supported datasets:
# - ml-100k
# - ml-1m
# - ml-10m

import tarfile
import requests
import zipfile
import io
import sys
import os


def download_and_extract_from_url(url, dataset_name):
    """
    Downloads and extracts the dataset from the given url
    and saves it in the data/dataset_name folder.

    Parameters:
        url (str): The url of the dataset.
        dataset_name (str): The name of the dataset.
    """

    print("Downloading and extracting " + dataset_name + " dataset...")
    r = requests.get(url)

    # If file is a zip file, extract it
    if zipfile.is_zipfile(io.BytesIO(r.content)):
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("data/")

    # If file is a tar file, extract it
    elif tarfile.is_tarfile(io.BytesIO(r.content)):
        z = tarfile.open(fileobj=io.BytesIO(r.content))
        z.extractall("data/")
    else:
        # Save the file
        os.makedirs("data/" + dataset_name, exist_ok=True)
        with open("data/" + dataset_name + "/training_test_dataset.mat", "wb") as f:
            f.write(r.content)

    print("Done!")


def download_data(dataset_name):
    """
    Calls the download_and_extract_from_url function
    with the correct url for the given dataset_name.

    Parameters:
        dataset_name (str): The name of the dataset.
    """

    if dataset_name == "ml-100k":
        download_and_extract_from_url("http://files.grouplens.org/datasets/movielens/ml-100k.zip", dataset_name)
    elif dataset_name == "ml-1m":
        download_and_extract_from_url("http://files.grouplens.org/datasets/movielens/ml-1m.zip", dataset_name)
    elif dataset_name == "ml-10m":
        download_and_extract_from_url("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dataset_name)
        os.rename("data/ml-10M100K", "data/ml-10m")
    elif dataset_name.startswith("tripadvisor"):
        city = dataset_name.split("-")[1]
        download_and_extract_from_url(f"https://zenodo.org/record/5644892/files/{city}.zip?download=1", dataset_name)
        os.rename(f"data/{city}", f"data/{dataset_name}")
    elif dataset_name == "douban-monti":
        download_and_extract_from_url(
            f"https://github.com/fmonti/mgcnn/blob/master/Data/douban/training_test_dataset.mat?raw=true", dataset_name
        )
    elif dataset_name == "netflix-prize":
        if not os.path.exists("data/netflix-prize"):
            download_and_extract_from_url(
            f"https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz", dataset_name
            )
            os.rename("data/download", "data/netflix-prize")

        # Untar training set
        if not os.path.exists("data/netflix-prize/ratings.csv"):
            print("Extracting training set...")
            tar = tarfile.open("data/netflix-prize/training_set.tar")
            tar.extractall("data/netflix-prize/")
            tar.close()

            # Merge training set files
            print("Merging training set files...")
            with open("data/netflix-prize/ratings.csv", "w") as outfile:
                outfile.write("user_id,movie_id,rating\n")
                for filename in os.listdir("data/netflix-prize/training_set"):
                    print(f"Merging {filename}...", end="\r")
                    if os.path.isfile(os.path.join("data/netflix-prize/training_set", filename)):
                        # Open it
                        with open(os.path.join("data/netflix-prize/training_set", filename)) as infile:
                            
                            # The first line is the movie id
                            movie_id = infile.readline().split(":")[0]

                            # The rest of the lines are the user id and rating
                            for line in infile:
                                user_id, rating, _ = line.split(",")
                                outfile.write(f"{user_id},{movie_id},{rating}\n")

            # Remove everything inside the training_set folder
            for filename in os.listdir("data/netflix-prize/training_set"):
                os.remove(os.path.join("data/netflix-prize/training_set", filename))
            os.rmdir("data/netflix-prize/training_set")

    else:
        print("Dataset not supported yet.")
        return

    print("Dataset " + dataset_name + " downloaded and extracted successfully.")


if __name__ == "__main__":
    # Read dataset name from command line
    if len(sys.argv) != 2:
        print("Usage: python download_data.py DATASET_NAME")
        sys.exit(1)

    dataset_name = sys.argv[1]

    download_data(dataset_name)
