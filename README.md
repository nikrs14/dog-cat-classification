# Dog or Cat classification
---
Convolutional Neural Network for a species classification using PyTorch

| | |
| --- | --- |
| Made in | October 2022 |
| Languages used | Python 3 |
| Libraries used | Pytorch, cv2, numpy, pandas|
| Programs used | VSCode |

## How to use
---
The images can be found on [this link](https://www.kaggle.com/datasets/tongpython/cat-and-dog). Note that the `data.csv` file only works if all the images are on the same directory (`./dataset/set`).

On `./main`, run `python3 train.py` to train the model, and run `python3 eval.py` to evaluate the average precision. On `./saves` there are data for pre-trained models (`model01.pth` with 99.99% average precision.)
