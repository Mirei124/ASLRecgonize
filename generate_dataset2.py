import numpy as np
from tqdm import tqdm

from dataset2 import ASLDatasetRaw

# dataset = ASLDatasetRaw(is_train=True)
dataset = ASLDatasetRaw(is_train=False)

x_list = []
y_list = []
for i, (data, label) in tqdm(enumerate(dataset)):
    x_list.append(data.numpy())
    y_list.append(label.item())


all_x = np.array(x_list)
all_y = np.array(y_list)

print(f"{all_x.shape=}")
print(f"{all_y.shape=}")

# np.save("train_x.npy", all_x)
# np.save("train_y.npy", all_y)
np.save("test_x.npy", all_x)
np.save("test_y.npy", all_y)
