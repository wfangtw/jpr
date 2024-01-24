# Credits: The design of the source code follows from the UPR and DPR download_data.py script


"""
 Command line tool to download various preprocessed data sources for UPR.
"""
import argparse
import tarfile
import gzip
import os
import pathlib
from subprocess import Popen, PIPE


NQ_LICENSE_FILES = [
    "https://dl.fbaipublicfiles.com/dpr/nq_license/LICENSE",
    "https://dl.fbaipublicfiles.com/dpr/nq_license/README",
]

RESOURCES_MAP = {
    # Wikipedia
    "data.wikipedia-split.psgs_w100": {
        "url": "https://www.dropbox.com/s/bezryc9win2bha1/psgs_w100.tar.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "compression_type": "tar.gz",
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },
    # DPR
    "data.retriever.biencoder-nq-train": {
        "url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "compression_type": "gz",
        "desc": "NQ train subset with passages pools for the Retriever training",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.biencoder-trivia-train": {
        "url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "compression_type": "gz",
        "desc": "TriviaQA train subset with passages pools for the Retriever training",
    },

    # BM25
    "data.retriever-outputs.bm25.nq-test": {
        "url": "https://www.dropbox.com/s/ml2lnt34ktjgft6/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "compression_type": "tar.gz",
        "desc": "Top-1000 passages from BM25 retriever for Natural Questions Open test set.",
    },
    "data.retriever-outputs.bm25.nq-dev": {
        "url": "https://www.dropbox.com/s/2gx8mwj58ifxwm2/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "compression_type": "tar.gz",
        "desc": "Top-1000 passages from BM25 retriever for Natural Questions Open development set.",
    },
    "data.retriever-outputs.bm25.trivia-dev": {
        "url": "https://www.dropbox.com/s/dd04rdrk85fj6kz/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "compression_type": "tar.gz",
        "desc": "Top-1000 passages from BM25 retriever for TriviaQA development set.",
    },
    "data.retriever-outputs.bm25.trivia-test": {
        "url": "https://www.dropbox.com/s/2cf4v77bay9cwnm/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "compression_type": "tar.gz",
        "desc": "Top-1000 passages from BM25 retriever for TriviaQA test set.",
    },
}


def unpack(compressed_file: str, out_file: str, compression_type: str):
    print("Uncompressing %s", compressed_file)
    if compression_type == "tar.gz":
        input = tarfile.open(compressed_file, "r:gz")
        input.extractall(os.path.dirname(out_file))
        input.close()
    else:
        input = gzip.GzipFile(compressed_file, "rb")
        s = input.read()
        input.close()
        output = open(out_file, "wb")
        output.write(s)
        output.close()
    print(" Saved to %s", out_file)


def download_resource(
    url: str, original_ext: str, compressed: bool, compression_type: str, resource_key: str, out_dir: str
) -> None:
    print("Requested resource from %s", url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    print("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, "downloads", *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    print("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        print("File already exist %s", local_file_uncompressed)
        return

    local_file = os.path.abspath(os.path.join(save_root, path_names[-1] + (f".{compression_type}" if compressed else original_ext)))

    process = Popen(['wget', url, '-O', local_file], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    # print(stderr.decode("utf-8"))
    print("Downloaded to %s", local_file)

    if compressed:
        # uncompressed_path = os.path.join(save_root, path_names[-1])
        uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
        unpack(local_file, uncompressed_file, compression_type)
        os.remove(local_file)
    return



def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        print("matched by prefix resources: %s", resources)
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            print("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    download_url = download_info["url"]

    if isinstance(download_url, list):
        for i, url in enumerate(download_url):
            download_resource(
                url,
                download_info["original_ext"],
                download_info["compressed"],
                download_info["compression_type"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
    else:
        download_resource(
            download_url,
            download_info["original_ext"],
            download_info["compressed"],
            download_info["compression_type"],
            resource_key,
            out_dir,
        )
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print("Resource key=%s  :  %s", k, v["desc"])


if __name__ == "__main__":
    main()
