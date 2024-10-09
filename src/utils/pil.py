import torch

from PIL import Image
from torchvision.transforms import functional as F

image = Image.open(r'D:\WorkStation\Projects\Deep-Learning\images\Maybach.png')
print(image.size)       # w, h

image = F.to_tensor(image)
print(image.shape)      # c, h, w
