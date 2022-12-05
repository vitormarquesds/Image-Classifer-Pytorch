import os
import argparse
import torch
from preprocessing import preprocess_data
from tools import read_imagenet_classnames, display_results, run_inference, parse_base64
from torchvision import models

parse = argparse.ArgumentParser(description = 'Trained Model Inference')
parse.add_argument('-tp', '--top-predictions', metavar='NUMPRED', default=5,
                   help = 'Predictions per Image')

if __name__ == "__main__":

    args = parse.parse_args()

    model = models.resnet18(pretrained = True)

    imagenet_classes = read_imagenet_classnames("cache/imagenet_classnames.txt")

    data = preprocess_data("cache")

    predictions = run_inference(model, data[0], int(args.top_predictions))

    display_results(data, predictions, imagenet_classes)