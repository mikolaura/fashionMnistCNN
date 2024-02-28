import random
import torch
import helper_function
import model
import dataset
import pandas as pd
import matplotlib.pyplot as plt


model_1 = model.CNN(input_shape=1,
                    hidden_units=16,
                    output_shape=10)

model_1.load_state_dict(torch.load("cnn_fashion_mnist_model_0.pth"))
# loss_fn = torch.nn.CrossEntropyLoss()
# model_1_results = helper_function.eval_model(accuracy_fn=helper_function.accuracy_fn,
#                            data_loader=dataset.test_dataloader,
#                            loss_fn=loss_fn,
#                            model=model_1)

# compare_results = pd.DataFrame([
#                                 model_1_results,
#                                 ])
# compare_results.style.format(precision=3, thousands=".", decimal=",").format_index(str.upper, axis=1)
# print(compare_results)
image, label = dataset.test_data[7898]
model_1.eval()
with torch.inference_mode():
    y_pred = dataset.classes_names[model_1(image.unsqueeze(0)).argmax()]
plt.imshow(image.permute(1,2,0), cmap="gray")
plt.title(label=y_pred)
plt.show()
print(f"It have to be {dataset.classes_names[label]} and it is {y_pred}")