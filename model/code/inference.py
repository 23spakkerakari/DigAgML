import tensorflow as tf
import json
import numpy as np

def model_fn(model_dir):
    model = tf.keras.models.load_model(f"{model_dir}/model.h5") #might need compile = false
    return model


def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data)
    raise Exception("Unsupported content type")


def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction


def output_fn(prediction, accept):
    return json.dumps(prediction.tolist()), accept