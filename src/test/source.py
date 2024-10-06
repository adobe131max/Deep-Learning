import torch
import torch.nn as nn

input = torch.tensor([[-1, 2, 3]])
print(torch.relu(input))
print(nn.functional.relu(input))

relu = nn.ReLU(inplace=True)
relu(input)
print(input)