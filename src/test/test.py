import torch
import numpy as np

keypoints = np.array([1, 2, 3, 4, 5, 6]).reshape([-1, 3])
print(keypoints)

visible = keypoints[:, 2]
keypoints = keypoints[:, :2]
print(visible)
print(keypoints)

print(np.array([1, 2, 3, 4, 5, 6]).reshape([-1, 3])[:, :2])

print('123'
      '456')

ml = [1, 2, 3]
for val in ml:
      val = val + 1
print(ml)

print(list((1, 2, 3, 4)))
print(list([1, 2, 3]))

print([1, 2, 3] * 2)
print([[1, 2, 3]] * 2)

nl = [ml] * 2
ml[0] = 0
print(nl)

anchors_over_all_feature_maps = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
anchors = []
for i in range(3):
      anchors_in_image = []
      for anchors_per_feature_map in anchors_over_all_feature_maps:
            anchors_in_image.append(anchors_per_feature_map)
      anchors.append(anchors_in_image)
anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
print(f'id(anchors[0]) == id(anchors[1]): {id(anchors[0]) == id(anchors[1])}')
anchors[0][0] = 0
print(anchors)

anchors_over_all_feature_maps = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
anchors = [torch.cat(anchors_over_all_feature_maps)] * 3
print(f'id(anchors[0]) == id(anchors[1]): {id(anchors[0]) == id(anchors[1])}')
anchors[0][0] = 0
print(anchors)

anchors_over_all_feature_maps = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
anchors = [anchors_over_all_feature_maps] * 3
anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
print(f'id(anchors[0]) == id(anchors[1]): {id(anchors[0]) == id(anchors[1])}')
anchors[0][0] = 0
print(anchors)

# 高级切片
obj = torch.tensor([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9 ,0]])
batch_idx = torch.tensor([[0],
                          [1]])
top_n_idx = torch.tensor([[1, 2, 0],
                          [4, 3, 2]])
print(obj[batch_idx, top_n_idx])

boxs = torch.tensor([[1, 2, 3, 4],
                     [5, 6, 7, 8]])
print(boxs[..., 0::2])
print(boxs[:, 0::2])
x = boxs[:, 0::2]
y = boxs[:, 1::2]
print(torch.cat((x, y), dim=1))
print(torch.stack((x, y), dim=1))
print(torch.stack((x, y), dim=2))

print(boxs.max())
