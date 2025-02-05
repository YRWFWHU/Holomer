import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from odak.learn.wave import band_limited_angular_spectrum, wavenumber, propagate_beam


class End2End(pl.LightningModule):
    def __init__(self,
                 slm_phase_generator: nn.Module,
                 loss_func=None,
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
        self.wavelengths = wavelengths
        self.distances = distances
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.slm_phase_predictor = slm_phase_generator
        self.lr = learning_rate
        if loss_func is None:
            self.loss = nn.MSELoss()
        else:
            self.loss = loss_func
        self.num_of_frames = num_of_frames
        self.complex = complex_model

        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

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
        slm_phase = self.slm_phase_predictor(target_amp)
        slm_complex = torch.complex(torch.cos(slm_phase), torch.sin(slm_phase))
        recon_amp = torch.abs(propagate_beam(slm_complex,
                                             wavenumber(self.wavelengths[0]),
                                             self.distances[0],
                                             self.pixel_size,
                                             self.wavelengths[0],
                                             zero_padding=[True, False, True],
                                             ))

        return recon_amp, slm_phase

    def training_step(self, batch, batch_idx):
        recon_amp, slm_phase = self.forward(batch)
        loss = self.loss(recon_amp, batch)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        recon_intensity = recon_amp ** 2
        self.log('Max Value of Reconstructed Image', recon_intensity.max())
        self.log('PSNR', self.psnr(recon_intensity[:, 0:1, :, :], batch), prog_bar=True, on_epoch=True)
        self.log('SSIM', self.ssim(recon_intensity[:, 0:1, :, :], batch), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        recon_amp, slm_phase = self.forward(batch)
        loss = self.loss(recon_amp, batch)
        self.log('validation_loss', loss)
        recon_intensity = recon_amp ** 2
        if batch_idx == 0:
            self.logger.experiment.add_image('Reconstructed Image', recon_intensity[0, 0:1, :, :],
                                             global_step=self.global_step)
        return loss

    def test_step(self, batch, batch_idx):
        recon_intensity, slm_phase = self.forward(batch)
        loss = self.loss(recon_intensity, batch)
        self.log('test_loss', loss)
        self.logger.experiment.add_image('Reconstructed Image', recon_intensity[0] / 1.414,
                                         global_step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
