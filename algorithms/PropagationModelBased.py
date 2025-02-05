import torch
from odak.learn.wave import get_band_limited_angular_spectrum_kernel, zero_pad, crop_center, custom


class PropagationModel(torch.nn.Module):
    def __init__(self, wavelengths, pixel_size, resolution, distances):
        """

        :param wavelengths: list, ordered in RGB
        :param pixel_size: float,
        :param resolution: list, H,W
        :param distances: list, multi-plane distance
        """
        super().__init__()
        self.wavelengths = wavelengths
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.distances = distances

        self.asm_kernel = torch.randn(
            len(self.wavelengths),
            len(self.distances),
            self.resolution[-2] * 2,  # *2 for zero padding
            self.resolution[-1] * 2,
            dtype=torch.complex64,
        )
        for idx_wave, wave in enumerate(self.wavelengths):
            for idx_dist, dist in enumerate(self.distances):
                self.asm_kernel[idx_wave, idx_dist] = get_band_limited_angular_spectrum_kernel(
                    resolution[-2] * 2, resolution[-1] * 2, dx=pixel_size, wavelength=wave, distance=dist)

    def reconstruct_intensity(self, hologram):
        """
        reconstruct hologram
        :param hologram: input hologram, shape is B×T×H×W, complex field
        :return: reconstructed img, shape is B×C×H×W×D
        """
        hologram = zero_pad(hologram)
        recon_intensity = torch.ones(
            hologram.shape[0],
            len(self.wavelengths),
            self.resolution[-2],
            self.resolution[-1],
            len(self.distances),
        )
        for idx_wave, wave in enumerate(self.wavelengths):
            for idx_dist, dist in enumerate(self.distances):
                # multi-frame complex: B×T×H×W
                recon_complex = crop_center(custom(hologram, self.asm_kernel[idx_wave, idx_dist], zero_padding=False,
                                                   aperture=1.))
                recon_intensity[:, idx_wave, :, :, idx_dist] = torch.sum(torch.abs(recon_complex) ** 2, dim=1)
        return recon_intensity

    def prop_field(self, source_field):
        """
        propagate a B×C×H×W field to some depth as B×C×H×W field
        :param source_field: source field , shape is B×T×H×W, complex field
        :return: reconstructed complex field, shape is B×C×H×W×D
        """

        device = source_field.device
        recon_field = torch.ones_like(source_field, dtype=torch.complex64).to(device)
        source_field = zero_pad(source_field)
        for idx_wave, wave in enumerate(self.wavelengths):
            recon_field[:, idx_wave, :, :] = crop_center(
                custom(source_field[:, idx_wave], self.asm_kernel[idx_wave, 0], zero_padding=False, aperture=1.))

        return recon_field

    def prop_field_keepdim(self, hologram):
        """
        each channel generate a single color reconstructed image
        :param hologram: B×4×H×W
        :return: B×4×H×W
        """
        hologram = zero_pad(hologram)
        recon_field = crop_center(custom(hologram, self.asm_kernel[0, 0], zero_padding=False, aperture=1.))
        return recon_field

    def forward(self, x):
        pass
