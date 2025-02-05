import torch
import torchmetrics
import torchvision.utils
from torchvision import transforms
from odak.learn.wave import generate_complex_field
from odak.learn.wave import band_limited_angular_spectrum, wavenumber, propagate_beam
import pytorch_lightning as pl
from PIL import Image


class SGD(pl.LightningModule):
    def __init__(self, wavelengths, pixel_size, resolution, distances, target_path, learning_rate, number_of_frame,
                 num):
        """
        :param wavelengths:
        :param pixel_size:
        :param resolution:
        :param distances:
        :param target_path: path to target, str shape is B×C×H×W×D
        :param learning_rate:
        :param number_of_frame:
        """
        torch.manual_seed(42)
        super().__init__()
        self.wavelengths = wavelengths
        self.pixel_size = pixel_size
        self.distances = distances
        data_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ])
        self.num = num
        self.num_of_frame = number_of_frame
        self.target = data_transform(Image.open(target_path)).unsqueeze(0)
        self.target_amp = torch.sqrt(self.target)
        # shape of phase: B×T×H×W
        self.phase = torch.randn(
            self.target.shape[0],
            number_of_frame,
            resolution[-2],
            resolution[-1],
            requires_grad=True
        )
        self.lr = learning_rate
        self.loss = torch.nn.MSELoss()
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    def training_step(self, batch):
        hologram = generate_complex_field(1., self.phase)

        recon_intensity = []
        for wave_idx, wavelength in enumerate(self.wavelengths):
            for frame_idx in range(self.num_of_frame):
                try:
                    recon_intensity[wave_idx] += torch.abs(propagate_beam(
                        hologram[:, frame_idx:frame_idx + 1, :, :],
                        wavenumber(wavelength),
                        self.distances[0],
                        self.pixel_size,
                        wavelength,
                        propagation_type='Transfer Function Fresnel',
                        zero_padding=[True, False, True],
                    )) ** 2
                except IndexError:
                    recon_intensity.append(torch.abs(propagate_beam(
                        hologram[:, frame_idx:frame_idx + 1, :, :],
                        wavenumber(wavelength),
                        self.distances[0],
                        self.pixel_size,
                        wavelength,
                        propagation_type='Transfer Function Fresnel',
                        zero_padding=[True, False, True],
                    )) ** 2)

        reconstruction = torch.cat(recon_intensity, dim=1)

        loss = self.loss(reconstruction, self.target)

        self.log('Max Value of Reconstructed Image', reconstruction.max())
        self.log('Min Value', reconstruction.min())
        self.log('MSE loss', loss, prog_bar=True)
        self.log('PSNR intensity', self.psnr(reconstruction, self.target), prog_bar=True)
        self.log('SSIM intensity', self.ssim(reconstruction, self.target), prog_bar=True)
        self.log('PSNR srgb', self.psnr(self.srgb_lin2gamma(reconstruction), self.srgb_lin2gamma(self.target)),
                 prog_bar=True)
        self.log('SSIM srgb', self.ssim(self.srgb_lin2gamma(reconstruction), self.srgb_lin2gamma(self.target)),
                 prog_bar=True)
        self.logger.experiment.add_image('Reconstructed Image', reconstruction[0] / 1.414, global_step=self.global_step)
        return loss

    def on_train_end(self):
        hologram = generate_complex_field(1., self.phase)

        recon_intensity = []
        for wave_idx, wavelength in enumerate(self.wavelengths):
            for frame_idx in range(self.num_of_frame):
                recon_intensity[wave_idx] += torch.abs(propagate_beam(
                    hologram[:, frame_idx:frame_idx + 1, :, :],
                    wavenumber(self.wavelengths[wave_idx]),
                    self.distances[0],
                    self.pixel_size,
                    self.wavelengths[wave_idx],
                    zero_padding=[True, False, True],
                )) ** 2

        reconstruction = torch.cat(recon_intensity, dim=1)
        # recon_amp_r = torch.abs(propagate_beam(hologram[:, 0:1, :, :],
        #                                        wavenumber(self.wavelengths[0]),
        #                                        self.distances[0],
        #                                        self.pixel_size,
        #                                        self.wavelengths[0],
        #                                        zero_padding=[True, False, True],
        #                                        ))
        # recon_amp_g = torch.abs(propagate_beam(hologram[:, 1:2, :, :],
        #                                        wavenumber(self.wavelengths[1]),
        #                                        self.distances[0],
        #                                        self.pixel_size,
        #                                        self.wavelengths[1],
        #                                        zero_padding=[True, False, True],
        #                                        ))
        # recon_amp_b = torch.abs(propagate_beam(hologram[:, 2:3, :, :],
        #                                        wavenumber(self.wavelengths[2]),
        #                                        self.distances[0],
        #                                        self.pixel_size,
        #                                        self.wavelengths[2],
        #                                        zero_padding=[True, False, True],
        #                                        ))
        # recon_amp = torch.cat([recon_amp_r, recon_amp_g, recon_amp_b], dim=1)
        # recon_amp = recon_amp * (
        #         torch.sum(recon_amp * self.target_amp, (-2, -1), keepdim=True) / torch.sum(recon_amp * recon_amp,
        #                                                                                    (-2, -1), keepdim=True))
        # reconstruction = recon_amp ** 2
        torchvision.utils.save_image(reconstruction, f'images/SGD/Recon_{self.num}.png', normalize=True)
        torchvision.utils.save_image(self.phase % (2 * torch.pi), f'images/SGD/Hologram_{self.num}.png', normalize=True)
        torchvision.utils.save_image(self.target, f'images/SGD/Target_{self.num}.png', normalize=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([self.phase], lr=self.lr)
        return optimizer

    @staticmethod
    def srgb_lin2gamma(im_in):
        """converts from linear to sRGB color space"""
        thresh = 0.0031308
        im_out = torch.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in ** (1 / 2.4)) - 0.055)
        return im_out
