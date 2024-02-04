import os
import subprocess

import pandas as pd
from google.cloud import storage


def generate_images(image_width: int):
    # get all the videos
    videos = [
        file for file in os.listdir("training_data/videos") if file.endswith(".mov")
    ]

    for video in videos:
        # label is the video filename
        label = video.removesuffix(".mov")
        os.mkdir(f"training_data/{label}")

        # extract images from the video
        result = subprocess.check_call(
            [  # Docker parameters for running ffmpeg
                "docker",
                "run",
                "-v",
                f"{os.getcwd()}/training_data/:/workdir",
                "-w",
                "/workdir",
                "--rm",
                "jrottenberg/ffmpeg",
            ]
            + [  # ffmpeg parameters
                "-i",
                f"videos/{video}",
                "-vf",
                "fps=10",  # Keep only 10 frames per second
                "-vf",
                f"scale={image_width}:-1",  # Output scaling
                f"{label}/%04d.jpg",
            ]
        )


def upload_training_data_to_gc(project: str, bucket_name: str, region: str):

    # Create the bucket
    storage_client = storage.Client(project=project)
    bucket = storage_client.create_bucket(bucket_name, project=project, location=region)
    bucket_uri = f"gs://{bucket_name}"
    print(f"Data bucket created: {bucket_uri}")

    # Array to keep reference to training data and label
    training_data = []

    for root, dirs, files in os.walk("training_data/"):
        # skip videos folder
        if root in ["training_data/", "training_data/videos"]:
            print(f"Skipping folder: {root}")
            continue

        print(f"Processing folder: {root}")
        for file in files:
            filename = os.path.join(root, file)
            blob = bucket.blob(filename)
            blob.upload_from_filename(filename)

            training_data.append(
                (f"{bucket_uri}/{filename}", root.removeprefix("training_data/"))
            )

    # The CSV will instruct AutoML which data to use to train the model
    training_assets_filename = "training_data/training_data.csv"
    dataframe = pd.DataFrame(data=training_data)
    dataframe.to_csv(training_assets_filename, index=False, header=False)

    blob = bucket.blob(training_assets_filename)
    blob.upload_from_filename(training_assets_filename)

    return bucket_uri
