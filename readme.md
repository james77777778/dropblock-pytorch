# Dropblock
Implementation of DropBlock: A regularization method for convolutional networks in PyTorch.
Follow the same logic from official implementation:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/modeling/architecture/nn_ops.py

## Usage
It is only for 2D input now.
```python
import torch
from dropblock import DropBlock2D

x = torch.rand(1, 3, 256, 256)

drop_block = DropBlock2D(dropblock_prop=0.2, dropblock_size=7)
regularized_x = drop_block(x)
```

## Implementation details
I strictly follow the same logic from official implementation but there are still some differences due to different framework in Tensorflow and PyTorch.

### `torch.meshgrid` needs additional `transpose`
The output of `torch.meshgrid` needs to be transposed to be the same as `tf.meshgrid`. This should be fixed.

### `'SAME'` padding in `F.max_pool2d`
The behavior of `F.max_pool2d` should be similar but not the same as `tf.nn.max_pool` with `padding='SAME'`.

## TODO
- [ ] Check the performance of DropBlock
- [ ] Test Unit of DropBlock
- [ ] Applied DropBlock in detection model (RetinaNet with SpineNet)