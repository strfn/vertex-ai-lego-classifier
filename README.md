# Mini figures classifier

Playground on using AutoML to classify lego mini figures.

Training data is generated extracting sample images from short videos of the mini figures.

To make things more interesting the videos have been taken in low light condition by a young child without any supervision or direction.
A few random items has been added to the training set (why not...)

![Minifigures video](/docs/stacked.gif)

## Prerequisites
- Docker
- pyenv (or any other preferred venv system)
- Google cloud cli
- Google cloud project
- Videos in `.mov` format of the mini figure (or anything else) to train the model with into the `training_data/videos/` folder

```shell
# Create and activate a virtual environment
pyenv virtualenv 3.12.1 lego_classifier
pyenv local lego_classifier

# Install dependencies
pip install pip-tools
pip-compile --upgrade requirements.piptools
pip install -r requirements.txt

# google account authentication
gcloud auth application-default login
```

## Run it !
The CLI automate the process of preparing the data set and then training the model. Parameters can be set as `ENV` variables when indicated or simply passed as options to the CLI command.

### Data preparation

```shell
./main.py data-prep --help

 Usage: main.py data-prep [OPTIONS]

 Generate training images from the videos an upload the dataset to a NEW Google Cloud Storage.

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --project            TEXT     Project ID hosting the training data [env var: LC_PROJECT_ID] [default: None] [required]                                       │
│    --img-width          INTEGER  Image width in pixels, height will maintain aspect ratio, [env var: LC_IMG_WIDTH] [default: 130]                               │
│    --bucket-name        TEXT     How to name the bucket for the training images. [env var: LC_BUCKET_NAME] [default: lcb-a34ade54e32b422e9c06d59f4bc61cd3]      │
│    --region             TEXT     The region in which to upload the training data. [env var: LC_REGION] [default: us-central1]                                   │
│    --help                        Show this message and exit.                                                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Model training and deployment

```shell
./main.py train-model --help

 Usage: main.py train-model [OPTIONS] TRAINING_SOURCE_BUCKET

 Train and deploy the image classifier with Vertex AI AutoML.

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    training_source_bucket      TEXT  Training bucker name [env var: LC_BUCKET_NAME] [default: None] [required]                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --project                    TEXT  Project ID [env var: LC_PROJECT_ID] [default: None] [required]                                                           │
│ *  --staging-bucket-name        TEXT  Staging bucket name [env var: LC_STAGING_BUCKET] [default: None] [required]                                              │
│    --model-type                 TEXT  Model type, see                                                                                                          │
│                                       https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.AutoMLImageTrainingJob          │
│                                       [default: CLOUD]                                                                                                         │
│    --region                     TEXT  The where to train and host the model. [env var: LC_REGION] [default: us-central1]                                       │
│    --help                             Show this message and exit.                                                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


```

### Image classification

```shell
Usage: main.py inference [OPTIONS] FILENAME

 Infer which mini figure is visible in an image.

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    filename      TEXT  File name of the image to classify. [default: None] [required]                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --endpoint-id        TEXT  Model inference endpoint id. [env var: LC_ENDPOINT] [default: None] [required]                                                   │
│ *  --project            TEXT  Project ID [env var: LC_PROJECT_ID] [default: None] [required]                                                                   │
│    --region             TEXT  Region where the inference service is hosted. [env var: LC_REGION] [default: us-central1]                                        │
│    --help                     Show this message and exit.                                                                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
