# Pretrained MobileNet model - TensorFlow

An example program and functions to use pretrained models in tensorflow. Using the model from - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

![alt text](https://github.com/Ashok93/Pretrained-model-with-TensorFlow/blob/master/static_imgs/mobile_net_google.gif "RPS Image")

### Dependencies
1. Tensorflow
2. OpenCV
3. numpy

Create a `models` directory and place the models inside it.
Create a `logs/graph` directory for graph log - for tensorboard.
Please note to modify the config.json file. For eg) the path to the model, the model name, input/output nodes etc..

Run `python pretrained_network.py --config_json config.json` to see the result.



