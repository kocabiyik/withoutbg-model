# global variables for the project

import sys

import torch
import yaml

DTYPE = torch.float32
EPSILON = sys.float_info.epsilon
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {DEVICE} device')

with open('configurations.yaml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

DIM = conf["dataset"]["dim"]
BATCH_SIZE = conf["hparam"]["batch_size"]
