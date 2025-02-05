import torch
import pytorch_lightning as pl
import torchmetrics
import torchvision
from complexPyTorch.complexLayers import ComplexConvTranspose2d, ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu

from algorithms.PropagationModelBased import PropagationModel


class CCNN(torch.nn.Module):
    def __init__(self, num_of_stages=4, input_dim_of_first_layer=1, output_dim_of_first_layer=4, output_channels=1):
        """
        CCNN module for Complex Convolution Neural Network, Accept input shape: B×C×H×W
        :param num_of_stages: stages of Unet
        :param input_dim_of_first_layer: input tensor's channel number, C
        :param output_dim_of_first_layer: first stage's output dimension
        """
        super().__init__()
        self.DownSample = self.down_sample_layers_generator(num_of_stages=num_of_stages,
                                                            input_dim=input_dim_of_first_layer,
                                                            output_dim=output_dim_of_first_layer)
        self.UpSample = self.up_sample_layers_generator(num_of_stages=num_of_stages,
                                                        input_dim=input_dim_of_first_layer,
                                                        output_dim=output_dim_of_first_layer)
        self.OutputProjection = ComplexConv2d(in_channels=input_dim_of_first_layer, out_channels=output_channels,
                                              kernel_size=3, stride=1, padding=1)
        self.num_of_stages = num_of_stages

    def forward(self, x):
        output_of_down = []
        for down_layer in self.DownSample:
            x = down_layer(x)
            x = complex_relu(x)
            output_of_down.append(x)

        output_of_down.pop()

        for idx, up_layer in enumerate(self.UpSample):
            x = up_layer(x)
            if idx != (self.num_of_stages - 1):
                x = complex_relu(x)
                x += output_of_down.pop()

        x = self.OutputProjection(x)

        predict_phase = torch.atan2(x.imag, x.real)
        return predict_phase

    @staticmethod
    def down_sample_layers_generator(num_of_stages=4, input_dim=1, output_dim=4):
        layers = [ComplexConv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1)]
        for i in range(num_of_stages - 1):
            layers.append(
                ComplexConv2d(in_channels=output_dim, out_channels=output_dim * 2, kernel_size=3, stride=2, padding=1)
            )
            output_dim *= 2
        return torch.nn.ModuleList(layers)

    @staticmethod
    def up_sample_layers_generator(num_of_stages=4, input_dim=1, output_dim=4):
        layers = [
            ComplexConvTranspose2d(in_channels=output_dim, out_channels=input_dim, kernel_size=4, stride=2, padding=1)
        ]
        for i in range(num_of_stages - 1):
            layers.append(
                ComplexConvTranspose2d(in_channels=output_dim * 2,
                                       out_channels=output_dim,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)
            )
            output_dim *= 2
        layers.reverse()
        return torch.nn.ModuleList(layers)

