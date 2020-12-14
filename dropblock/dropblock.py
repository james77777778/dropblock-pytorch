import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """ DropBlock: a regularization method for convolutional neural networks.

        DropBlock is a form of structured dropout, where units in a contiguous
        region of a feature map are dropped together. DropBlock works better than
        dropout on convolutional layers due to the fact that activation units in
        convolutional layers are spatially correlated.
        See https://arxiv.org/pdf/1810.12890.pdf for details.

        Modified from:
        https://github.com/tensorflow/tpu/blob/master/models/official/detection/modeling/architecture/nn_ops.py#L191
    """
    def __init__(
        self,
        dropblock_prop=None,
        dropblock_size=None,
    ):
        super(DropBlock2D, self).__init__()
        self._dropblock_keep_prob = 1.0 - dropblock_prop
        self._dropblock_size = dropblock_size

    def forward(self, x):
        """Builds Dropblock layer.
        Args:
            x: `Tensor` input tensor (bsize, channels, height, width).
        Returns:
            A version of input tensor with DropBlock applied.
        """
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self._dropblock_keep_prob == 1.0:
            return x

        # batch, channel, height, width
        _, _, height, width = x.size()
        total_size = width * height
        dropblock_size = min(self._dropblock_size, min(width, height))

        # seed_drop_rate is the gamma parameter of DropBlcok.
        seed_drop_rate = self._compute_gamma(total_size, dropblock_size, width, height)

        # Forces the block to be inside the feature map.
        w_i, h_i = torch.meshgrid(torch.range(0, width - 1), torch.range(0, height - 1))
        w_i, h_i = w_i.transpose(0, 1), h_i.transpose(0, 1)

        valid_block = torch.logical_and(
            torch.logical_and(w_i >= int(dropblock_size // 2), w_i < width - (dropblock_size - 1) // 2),
            torch.logical_and(h_i >= int(dropblock_size // 2), h_i < width - (dropblock_size - 1) // 2))
        valid_block = valid_block.float()
        valid_block = torch.reshape(valid_block, [1, 1, height, width])

        randnoise = torch.rand(x.shape, dtype=torch.float32)
        seed_keep_rate = 1.0 - seed_drop_rate
        block_pattern = ((1 - valid_block + seed_keep_rate + randnoise) >= 1).float()

        block_pattern = -F.max_pool2d(
            -block_pattern,
            kernel_size=(self._dropblock_size, self._dropblock_size),
            stride=(1, 1),
            padding=self._dropblock_size // 2
        )

        percent_ones = (block_pattern.sum() / block_pattern.numel()).float()
        percent_ones = percent_ones.to(x.device)
        block_pattern = block_pattern.to(x.device)

        x = x * block_pattern / percent_ones
        return x

    def _compute_gamma(self, total_size, dropblock_size, width, height):
        seed_drop_rate = (
            1.0 - self._dropblock_keep_prob) * total_size / dropblock_size**2 / (
                (width - self._dropblock_size + 1)
                * (height - self._dropblock_size + 1)
        )
        return seed_drop_rate
