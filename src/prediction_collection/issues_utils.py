import os
import tarfile
import urllib.request

DOWNLOAD_URL_TRAINING = "https://tickettagger.blob.core.windows.net/datasets/github-labels-top3-803k-train.tar.gz"
DOWNLOAD_URL_TEST = "https://tickettagger.blob.core.windows.net/datasets/github-labels-top3-803k-test.tar.gz"

UNPACKED_DATA_DIR = "/generated/datasets/"

def prepare_datasets():
    if not os.path.exists(UNPACKED_DATA_DIR):
        os.makedirs(UNPACKED_DATA_DIR)

    if os.path.exists(f"{UNPACKED_DATA_DIR}/issues-training"):
        print("Training dataset already downloaded and unpacked.")
    else:
        print("Downloading training dataset...")
        training_tar_gz = "/tmp/ticket-tagger-training.tar.gz"
        urllib.request.urlretrieve(DOWNLOAD_URL_TRAINING, training_tar_gz)
        tar = tarfile.open(training_tar_gz, "r:gz")
        tar.extractall(path=f"{UNPACKED_DATA_DIR}/issues-training")
        tar.close()
        print("Training dataset downloaded and unpacked.")

    if os.path.exists(f"{UNPACKED_DATA_DIR}/issues-test"):
        print("Test dataset already downloaded and unpacked.")
    else:
        print("Downloading test dataset...")
        test_tar_gz = "/tmp/ticket-tagger-test.tar.gz"
        urllib.request.urlretrieve(DOWNLOAD_URL_TEST, test_tar_gz)
        tar = tarfile.open(test_tar_gz, "r:gz")
        tar.extractall(path=f"{UNPACKED_DATA_DIR}/issues-test")
        tar.close()
        print("Test dataset downloaded and unpacked.")


if __name__ == '__main__':
    prepare_datasets()
