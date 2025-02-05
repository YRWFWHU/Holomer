from Swin import BasicLayer
from torch import nn


class SingleLayerSwin(nn.Module):
    def __init__(self, resolution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = BasicLayer(
            dim=1,
            input_resolution=(resolution[0], resolution[1]),
            depth=2,
            num_heads=3,
            window_size=(16, 16),
            mlp_ratio=4.,
            qkv_bias=True,  qk_scale=None,
            drop=0., attn_drop=0.,
            norm_layer=nn.LayerNorm,
            downsample=None
        )


class SingleLayerCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    net = SingleLayerSwin(resolution=(1024, 1920))
    x = torch.rand((1024, 1920))
    y = net(x)
    print(y)
