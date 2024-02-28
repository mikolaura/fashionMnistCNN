import torch
import torch.nn as nn
import dataset
import matplotlib.pyplot as plt
import helper_function
import model
import training
from timeit import default_timer as timer

#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
#Set manual seed
torch.manual_seed(42)
# Creating an instance of model
model_0 = model.CNN(input_shape=1,
                    hidden_units=16,
                    output_shape=len(dataset.classes_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)
start_timer = timer()
training.training(model=model_0,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    train_dataloader=dataset.train_dataloader,
                    test_dataloader=dataset.test_dataloader,
                    epochs=3,
                    device=device)
end_timer = timer()
helper_function.print_train_time(start=start_timer,
                                 end=end_timer,
                                 device=device)
torch.save(model_0.state_dict(), "cnn_fashion_mnist_model_0.pth")