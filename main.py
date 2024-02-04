#!/usr/bin/env python3
import os.path
import uuid
from typing import Annotated

import google.cloud.aiplatform as aiplatform
import typer

import lfc.training as training

app = typer.Typer()


@app.command(
    help="Generate training images from the videos an upload the dataset to a NEW Google Cloud Storage.",
    short_help="Training data generation.",
)
def data_prep(
    project: Annotated[
        str,
        typer.Option(
            envvar="LC_PROJECT_ID",
            prompt="Google Project ID",
            help="Project ID hosting the training data",
        ),
    ],
    img_width: Annotated[
        int,
        typer.Option(
            envvar="LC_IMG_WIDTH",
            help="Image width in pixels, height will maintain aspect ratio,",
        ),
    ] = 130,
    bucket_name: Annotated[
        str,
        typer.Option(
            envvar="LC_BUCKET_NAME",
            help="How to name the bucket for the training images.",
        ),
    ] = "lcb-"
    + uuid.uuid4().hex,
    region: Annotated[
        str,
        typer.Option(
            envvar="LC_REGION", help="The region in which to upload the training data."
        ),
    ] = "us-central1",
):
    print("..... generating images")
    training.generate_images(img_width)

    print("..... uploading images to GC")
    gcs = training.upload_training_data_to_gc(
        project=project, bucket_name=bucket_name, region=region
    )

    print("..............................................")
    print("............TRAINING DATASET READY............")
    print("..............................................")
    print(f"\n Available at: {gcs}")


@app.command()
def train_model(
    project: Annotated[
        str,
        typer.Option(
            envvar="LC_PROJECT_ID",
            prompt="Google Project ID",
            help="Project ID hosting the training data",
        ),
    ],
    staging_bucket_name: Annotated[str, typer.Option()],
    training_source_bucket: Annotated[str, typer.Argument()],
    region: Annotated[
        str,
        typer.Option(
            envvar="LC_REGION", help="The region in which to upload the training data."
        ),
    ] = "us-central1",
):

    aiplatform.init(
        project=project,
        staging_bucket=staging_bucket_name,
        location="us-central1",
        experiment="lego-classifier",
        experiment_description="Using AutoML to identify lego figures",
    )
    print(f"Vertex SDK initialised: {aiplatform.aiplatform_version}")

    print(os.path.join(training_source_bucket, "training_data/training_data.csv"))
    exit(1)

    dataset = aiplatform.ImageDataset.create(
        display_name="Lego training",
        gcs_source=[
            os.path.join(training_source_bucket, "training_data/training_data.csv")
        ],
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
        create_request_timeout=60 * 60,  # give it an hour to import the dataset.
    )

    print(f"Training dataset created: {dataset.resource_name}")

    job = aiplatform.AutoMLImageTrainingJob(
        display_name="train-lego-figures-classifier-1",
        prediction_type="classification",
        multi_label=False,
        model_type="CLOUD",
        base_model=None,
    )

    print(f"Training job created: {job}")

    model = job.run(
        dataset=dataset,
        model_display_name="lego-figures-classifier",
        training_fraction_split=0.6,
        validation_fraction_split=0.2,
        test_fraction_split=0.2,
        budget_milli_node_hours=8000,
        disable_early_stopping=False,
        create_request_timeout=2 * 60 * 60,  # 2 hours to complete the training
    )

    print(".....................................")
    print("............MODEL CREATED............")
    print(".....................................")

    print(model.resource_name)

    endpoint = model.deploy()

    print("......................................")
    print("............MODEL DEPLOYED............")
    print("......................................")
    print(f"\n reachable at: {endpoint}")


if __name__ == "__main__":
    app()