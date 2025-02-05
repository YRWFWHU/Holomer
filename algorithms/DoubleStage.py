import torch
import torchvision.utils
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from torchvision import models
from odak.learn.wave import band_limited_angular_spectrum, wavenumber, propagate_beam
import os


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.layers = [0, 5, 10, 19, 28]  # 选择VGG网络中的一些层作为感知特征
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]  # 每一层的权重
        self.vgg.eval()  # 设置为评估模式
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        loss = 0.0
        for i in range(len(self.layers)):
            x_features = self.vgg[:self.layers[i] + 1](x)
            y_features = self.vgg[:self.layers[i] + 1](y)
            loss += self.weights[i] * self.loss(x_features, y_features)  # 计算L1距离
        return loss


class DoubleStage(pl.LightningModule):
    def __init__(self,
                 init_phase_generator: nn.Module,
                 slm_phase_generator: nn.Module,
                 wavelengths: list = (638e-9, 520e-9, 450e-9),
                 pixel_size: float = 6.4e-6,
                 resolution: list = (1024, 2048),
                 distances: list = [20e-2],
                 learning_rate: float = 1e-3,
                 num_of_frames: int = 3,
                 complex_model: bool = True):
        """
        Target image shape:     B×C×H×W

        Output hologram shape:  B×T×H×W

        :param init_phase_generator: in_ch = C , out_ch = C
        :param slm_phase_generator: in_ch = C, out_ch = T
        :param wavelengths: Propagation model's wavelengths, list
        :param pixel_size: Pixel size of SLM
        :param resolution: Resolution of SLM
        :param distances: Distance between SLM and Reconstructed image plane
        :param learning_rate: Learning rate of Network
        :param num_of_frames: Numbers of output hologram frames: T
        """
        super().__init__()
        torch.set_float32_matmul_precision('medium')

        self.wavelengths = wavelengths
        self.distances = distances
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.init_phase_generator = init_phase_generator
        self.slm_phase_predictor = slm_phase_generator
        self.lr = learning_rate
        self.loss = nn.MSELoss()

        self.num_of_frames = num_of_frames
        self.complex = complex_model

        self.example_input_array = torch.rand(4, len(wavelengths), resolution[0], resolution[1])

        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

        if self.wavelengths == [450e-9]:
            self.color_channel = 2
        elif self.wavelengths == [520e-9]:
            self.color_channel = 1
        else:
            self.color_channel = 0

    def forward(self, target_intensity):
        """
        Generate phase-only hologram according to input target intensity

        :param target_intensity: shape is B×C×H×W
        :return: [predicted hologram, initial phase]
            slm_phase : final phase only hologram, shape: B×T×H×W, T is the number of frames
            initial phase: predicted phase at target plane, shape: B×C×H×W
        """
        # shape of target_amp: B×C×H×W
        target_amp = torch.sqrt(target_intensity)
        if self.complex:
            rand_phase = torch.zeros_like(target_amp)
            init_phase_generator_input = torch.complex(target_amp * torch.cos(rand_phase),
                                                       target_amp * torch.sin(rand_phase))
        else:
            init_phase_generator_input = target_amp
        init_phase = self.init_phase_generator(init_phase_generator_input)
        init_complex = torch.complex(target_amp * torch.cos(init_phase), target_amp * torch.sin(init_phase))
        init_slm_complex = propagate_beam(init_complex,
                                          wavenumber(self.wavelengths[0]),
                                          -1 * self.distances[0],
                                          self.pixel_size,
                                          self.wavelengths[0],
                                          zero_padding=[True, False, True],
                                          )

        if self.complex:
            slm_phase_generator_input = init_slm_complex
        else:
            # amp and angle concatenate to B×2×H×W
            slm_phase_generator_input = torch.cat((torch.abs(init_slm_complex), torch.angle(init_slm_complex)), dim=1)

        slm_phase = self.slm_phase_predictor(slm_phase_generator_input)

        slm_complex = torch.complex(torch.cos(slm_phase), torch.sin(slm_phase))
        recon_amp = torch.abs(propagate_beam(slm_complex,
                                             wavenumber(self.wavelengths[0]),
                                             self.distances[0],
                                             self.pixel_size,
                                             self.wavelengths[0],
                                             zero_padding=[True, False, True],
                                             ))

        return recon_amp, slm_phase, init_phase

    def training_step(self, batch, batch_idx):
        target_amp = torch.sqrt(batch)
        recon_amp, slm_phase, init_phase = self.forward(batch)
        loss = self.loss(recon_amp, torch.sqrt(batch))
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        recon_amp = recon_amp * (
                torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True) / torch.sum(recon_amp * recon_amp,
                                                                                      (-2, -1), keepdim=True))
        recon_intensity = recon_amp ** 2
        self.log('Max Value of Reconstructed Image', recon_intensity.max(), sync_dist=True)
        self.log('PSNR amp', self.psnr(recon_amp, target_amp), on_epoch=True, on_step=False, sync_dist=True)
        self.log('SSIM amp', self.ssim(recon_amp, target_amp), on_epoch=True, on_step=False, sync_dist=True)
        self.log('PSNR intensity', self.psnr(recon_intensity, batch), on_epoch=True, on_step=False, sync_dist=True)
        self.log('SSIM intensity', self.ssim(recon_intensity, batch), on_epoch=True, on_step=False, sync_dist=True)
        self.log('PSNR srgb', self.psnr(self.srgb_lin2gamma(recon_intensity), self.srgb_lin2gamma(batch)),
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log('SSIM srgb', self.ssim(self.srgb_lin2gamma(recon_intensity), self.srgb_lin2gamma(batch)),
                 on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_amp = torch.sqrt(batch)
        recon_amp, slm_phase, init_phase = self.forward(batch)
        loss = self.loss(recon_amp, torch.sqrt(batch))
        self.log('valid_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        recon_amp = recon_amp * (
                torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True) / torch.sum(recon_amp * recon_amp,
                                                                                      (-2, -1), keepdim=True))
        recon_intensity = recon_amp ** 2
        self.log('Max Value of Reconstructed Image', recon_intensity.max(), sync_dist=True)
        self.log('PSNR amp', self.psnr(recon_amp, target_amp), on_epoch=True, on_step=False, sync_dist=True)
        self.log('SSIM amp', self.ssim(recon_amp, target_amp), on_epoch=True, on_step=False, sync_dist=True)
        self.log('PSNR intensity', self.psnr(recon_intensity, batch), on_epoch=True, on_step=False, sync_dist=True)
        self.log('SSIM intensity', self.ssim(recon_intensity, batch), on_epoch=True, on_step=False, sync_dist=True)
        self.log('PSNR srgb', self.psnr(self.srgb_lin2gamma(recon_intensity), self.srgb_lin2gamma(batch)),
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log('SSIM srgb', self.ssim(self.srgb_lin2gamma(recon_intensity), self.srgb_lin2gamma(batch)),
                 on_epoch=True, on_step=False, sync_dist=True)
        if batch_idx == 0:
            self.logger.experiment.add_image('Reconstructed Image', recon_intensity[0, 0:1, :, :],
                                             global_step=self.global_step)
        return loss

    def predict_step(self, batch, batch_idx):
        target_amp = torch.sqrt(batch)
        recon_amp, slm_phase, init_phase = self.forward(batch)
        loss = self.loss(recon_amp, torch.sqrt(batch))
        recon_amp = recon_amp * (
                torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True) / torch.sum(recon_amp * recon_amp,
                                                                                      (-2, -1), keepdim=True))
        recon_intensity = recon_amp ** 2
        torchvision.utils.save_image(recon_intensity, f'images/SwinUnet/Recon_{self.global_step}.png')
        torchvision.utils.save_image(slm_phase, f'images/SwinUnet/Hologram_{self.global_step}.png')

        msg = f'{self.global_step}th img:\n' \
              f'PSNR amp: {self.psnr(recon_amp, target_amp)}{os.linesep}' \
              f'SSIM amp: {self.ssim(recon_amp, target_amp)}{os.linesep}' \
              f'PSNR intensity: {self.psnr(recon_intensity, batch)}{os.linesep}' \
              f'SSIM intensity: {self.psnr(recon_intensity, batch)}{os.linesep}' \
              f'PSNR srgb: {self.psnr(self.srgb_lin2gamma(recon_intensity), self.srgb_lin2gamma(batch))}{os.linesep}' \
              f'SSIM srgb: {self.ssim(self.srgb_lin2gamma(recon_intensity), self.srgb_lin2gamma(batch))}{os.linesep}'
        return msg

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def srgb_lin2gamma(im_in):
        """converts from linear to sRGB color space"""
        thresh = 0.0031308
        im_out = torch.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in ** (1 / 2.4)) - 0.055)
        return im_out

    @staticmethod
    def tv_loss(x):
        batch_size, channels, height, width = x.size()

        # 计算水平方向梯度
        h_tv = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])

        # 计算垂直方向梯度
        v_tv = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])

        # 计算总变差损失
        tv_loss = torch.sum(h_tv) + torch.sum(v_tv)

        # 归一化
        tv_loss = tv_loss / (batch_size * channels * height * width)
        return tv_loss
