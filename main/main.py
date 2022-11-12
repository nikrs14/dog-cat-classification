"""
Script for evaluating
a single image and
plot it
"""

import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sys import argv
from meta import path, device, transform
from convnet import ConvNet

def main():
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(path + f'/saves/{argv[2]}'))
    
    image_path = path + f'/images/{argv[1]}'
    plot_image = mpimg.imread(image_path)
    image = cv2.imread(image_path)
    image = transform(image)
    image = image.expand(1, 3, 64, 64)

    soft = nn.Softmax(dim=1)

    with torch.no_grad():
        image = torch.Tensor(image)
        image = image.to(device)
        output = model(image)
        weighted_output = soft(output)
        _, predicted = torch.max(output, 1)
        specie = 'dog' if predicted[0] == 0 else 'cat'
        precision = weighted_output[0][0] * 100 if predicted[0] == 0 else weighted_output[0][1] * 100
    
    imgplot = plt.imshow(plot_image)
    plt.xlabel(f'{specie} ({precision:.2f})%')
    plt.show()


if __name__ == '__main__':
    main()
