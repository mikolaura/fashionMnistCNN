import torch
import torch.nn as nn
from tqdm.auto import tqdm
import helper_function
def training(model: nn.Module,
             optimizer,
             loss_fn,
             train_dataloader,
             test_dataloader,
             epochs,
             device):
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        helper_function.train_step(model=model,
                                   dataloader=train_dataloader,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   device=device)
        helper_function.test_step(model=model,
                                  dataloader=test_dataloader,
                                  loss_fn=loss_fn,
                                  device=device)
