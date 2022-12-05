import base64
import re
import torch.nn.functional as F

def read_imagenet_classnames(path:str):

    with open(path, "r") as f:
        temp = f.readlines()

    temp = [i.strip(" ").strip("\n").strip(", ").split(":") for i in temp]
    temp = {int(k):v.strip(" ").split(",") for k,v in temp}

    classs = []

    for i in temp:
        classs.append([k.strip(" ") for k in temp[i]])
    return classs

def display_results(data, predictions, imagenet_classes, print_values = True):

    input_data, file_names = data

    probabilities, pred_index = predictions
    assert len(input_data) == len(file_names) == len(probabilities) == len(pred_index), "Checking..."

    pred_outputs = {}

    for i, file in enumerate(file_names):
        prob, idx = probabilities[i], pred_index[i]
        prediction = [f"{imagenet_classes[idx[j]][0]} with probability of {prob[j]:2f}%" for j in range(len(prob))]
        prediction = "\n".join(prediction)

        if print_values:
            print("\n---------------------------------")
            print("\nMaking Predictions!")
            print(f"\n File {file} prediction {prediction}")
        else:
            pred_outputs[file] = prediction
            if i == len(file_names) - 1:
                return pred_outputs

def one_prediction(prediction, imagenet_classes):

    prob, idx = prediction[0][0], prediction[1][0]

    return [{"class": str(imagenet_classes[idx[j]][0]).strip("'"), "probability": str(prob[j]).strip("'")} for j in range(len(prob))]

def run_inference(model, input_data, top_predictions):

    predictions = model(input_data)

    probabilities, pred_index = F.softmax(predictions,1).topk(top_predictions)

    probabilities = (probabilities * 100).detach().numpy()

    pred_index = pred_index.detach().numpy()

    return probabilities, pred_index

