import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
train_data = datasets.FashionMNIST(root="data",
                                   train=True,
                                   download=True,
                                   transform=ToTensor(),
                                   target_transform=None)
test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=ToTensor(),
                                  target_transform=None)
BATCH_SIZE = 32
train_dataloader =DataLoader(batch_size=BATCH_SIZE, dataset=train_data, shuffle=True)
test_dataloader = DataLoader(batch_size=BATCH_SIZE, dataset=test_data, shuffle=False)
classes_names = train_data.classes