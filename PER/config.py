import torch

ENV_NAME = 'MountainCar-v0'
gamma = 0.99
BATCH_SIZE = 128
LEARNING_RATE= 0.0006
BUFFER_SIZE = 8192
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
