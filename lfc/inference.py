import base64

from google.cloud.aiplatform.gapic.schema import predict


def get_inference_paramaters(
    filename: str, threshold: float = 0.7, predictions: int = 2
) -> (dict, dict):
    # read and convert the image to b64
    with open(filename, "rb") as f:
        file_content = f.read()
    encoded_content = base64.b64encode(file_content).decode("utf-8")

    # the instance to classify
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]

    # classification config
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=threshold,
        max_predictions=predictions,
    ).to_value()

    return instances, parameters
