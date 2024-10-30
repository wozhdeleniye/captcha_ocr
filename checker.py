from os import listdir

import torch
from torchvision.transforms import transforms
from image_preprocessing import preprocess_image
from PIL import Image
import random

from model import CaptchaModel, predict_symbol

transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
files = [(i[:-4], f'./generated/{i}') for i in listdir('./generated')]
random.shuffle(files)
model = CaptchaModel(num_classes=62)
model.load_state_dict(torch.load('captcha_model.pth'))
model.eval()
correct = 0
incorrect = dict()
for name, path in files:
    try:
        image_tensors = [transform(Image.fromarray(i)) for i in preprocess_image(path)]
        predictions = []
        for tensor in image_tensors:
            predictions.append(predict_symbol(model, tensor))
        if ''.join(predictions) == name:
            correct += 1
        print(''.join(predictions), name)
        if len(predictions) != len(name):
            continue
        for i in range(len(predictions)):
            if predictions[i] != name[i]:
                if name[i] not in incorrect.keys():
                    incorrect[name[i]] = 1
                else:
                    incorrect[name[i]] += 1
    except Exception as e:
        print(e)
print('Accuracy: ', correct / len(files))
