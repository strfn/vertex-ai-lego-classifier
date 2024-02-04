# Lego classifier

Playground on using AutoML to classify lego minifigures.

Training data is generated extracting sample images from short videos taken of the minifigures.
to avoid making the life to easy for the model the video has been taken in low light condition by a young child without any supervision or direction.
A few random items has been added to the training set (why not...)

![Minifigures video](/docs/stacked.gif)

## Prerequisites
- Docker
- pyenv (or any other preferred venv system)
- Google cloud cli
- Google cloud project

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
The CLI automate the process of preparing the data set and then training the model. Paramaters can be set as `ENV` variables when indicated or simply passed as options to the CLI command.

### Data preparation

```shell
./main.py data-prep --help

 Usage: main.py data-prep [OPTIONS]

 Generate training images from the videos an upload the dataset to a NEW Google Cloud Storage.

╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --project            TEXT     Project ID hosting the training data [env var: LC_PROJECT_ID] [default: None] [required]                                                 │
│    --img-width          INTEGER  Image width in pixels, height will maintain aspect ratio, [env var: LC_IMG_WIDTH] [default: 130]                                         │
│    --bucket-name        TEXT     How to name the bucket for the training images. [env var: LC_BUCKET_NAME] [default: lcb-a34ade54e32b422e9c06d59f4bc61cd3]                │
│    --region             TEXT     The region in which to upload the training data. [env var: LC_REGION] [default: us-central1]                                             │
│    --help                        Show this message and exit.                                                                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
