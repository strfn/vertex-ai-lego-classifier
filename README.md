# Title

short description

## Prerequisites
- Docker
- pyenv (or any other preferred venv system)
- Google cloud cli
- Google cloud project
- Authenticated account ``



```bash
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
