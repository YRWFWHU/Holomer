from algorithms.DoubleStage import DoubleStage
from algorithms.SGD import SGD
from algorithms.backbone.Swin import SwinTransformerSys
from algorithms.backbone.Unet import Unet
from algorithms.backbone.DualNet import CCNN
from dataset.DualNet import DIV2K, Div2k
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import time

# SwinUnet/Unet/CCNN
model = 'SwinUnet'
device = 'cuda:0'


def srgb_lin2gamma(im_in):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    im_out = torch.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in ** (1 / 2.4)) - 0.055)
    return im_out


ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

if model == 'SwinUnet':
    net_r = DoubleStage(
        init_phase_generator=SwinTransformerSys(img_size=[512, 512], patch_size=[4, 4], window_size=[16, 16],
                                                in_chans=1),
        slm_phase_generator=SwinTransformerSys(img_size=[512, 512], patch_size=[4, 4], window_size=[16, 16],
                                               in_chans=2),
        num_of_frames=1,
        complex_model=False,
        wavelengths=[638e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    net_g = DoubleStage(
        init_phase_generator=SwinTransformerSys(img_size=[512, 512], patch_size=[4, 4], window_size=[16, 16],
                                                in_chans=1),
        slm_phase_generator=SwinTransformerSys(img_size=[512, 512], patch_size=[4, 4], window_size=[16, 16],
                                               in_chans=2),
        num_of_frames=1,
        complex_model=False,
        wavelengths=[520e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    net_b = DoubleStage(
        init_phase_generator=SwinTransformerSys(img_size=[512, 512], patch_size=[4, 4], window_size=[16, 16],
                                                in_chans=1),
        slm_phase_generator=SwinTransformerSys(img_size=[512, 512], patch_size=[4, 4], window_size=[16, 16],
                                               in_chans=2),
        num_of_frames=1,
        complex_model=False,
        wavelengths=[450e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    checkpoint_r = torch.load(
        '/dev/shm/lightning_logs/SwinUnet_red_channel_window_size_16_WithoutTVLoss/version_0/checkpoints/epoch=141-step=113600.ckpt')
    net_r.load_state_dict(checkpoint_r['state_dict'])
    checkpoint_g = torch.load(
        '/dev/shm/lightning_logs/SwinUnet_green_channel_window_size_16_WithoutTVLoss/version_2/checkpoints/epoch=145-step=116800.ckpt')
    net_g.load_state_dict(checkpoint_g['state_dict'])
    checkpoint_b = torch.load(
        '/dev/shm/lightning_logs/SwinUnet_blue_channel_window_size_16_WithoutTVLoss/version_0/checkpoints/epoch=118-step=47600.ckpt')
    net_b.load_state_dict(checkpoint_b['state_dict'])
    resolution = (512, 512)
    model = 'SwinHolo_window16'

elif model == 'Unet':
    net_r = DoubleStage(
        init_phase_generator=Unet(in_channels=1, out_channels=1, nf0=32, num_down=7, max_channels=256,
                                  use_dropout=False, outermost_linear=True),
        slm_phase_generator=Unet(in_channels=2, out_channels=1, nf0=32, num_down=7, max_channels=256, use_dropout=False,
                                 outermost_linear=True),
        num_of_frames=1,
        complex_model=False,
        wavelengths=[638e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    net_g = DoubleStage(
        init_phase_generator=Unet(in_channels=1, out_channels=1, nf0=32, num_down=7, max_channels=256,
                                  use_dropout=False, outermost_linear=True),
        slm_phase_generator=Unet(in_channels=2, out_channels=1, nf0=32, num_down=7, max_channels=256, use_dropout=False,
                                 outermost_linear=True),
        num_of_frames=1,
        complex_model=False,
        wavelengths=[520e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    net_b = DoubleStage(
        init_phase_generator=Unet(in_channels=1, out_channels=1, nf0=32, num_down=7, max_channels=256,
                                  use_dropout=False, outermost_linear=True),
        slm_phase_generator=Unet(in_channels=2, out_channels=1, nf0=32, num_down=7, max_channels=256, use_dropout=False,
                                 outermost_linear=True),
        num_of_frames=1,
        complex_model=False,
        wavelengths=[450e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    checkpoint_r = torch.load(
        '/home/machine1/data/lightning_logs/Unet_red_1024_1920/version_0/checkpoints/epoch=209-step=168000.ckpt')
    net_r.load_state_dict(checkpoint_r['state_dict'])
    checkpoint_g = torch.load(
        '/home/machine1/data/lightning_logs/Unet_green_1024_1920/version_0/checkpoints/epoch=204-step=164000.ckpt')
    net_g.load_state_dict(checkpoint_g['state_dict'])
    checkpoint_b = torch.load(
        '/home/machine1/data/lightning_logs/Unet_blue_1024_1920/version_0/checkpoints/epoch=794-step=636000.ckpt')
    net_b.load_state_dict(checkpoint_b['state_dict'])
    resolution = (1024, 1920)

elif model == 'CCNN':
    net_r = DoubleStage(
        init_phase_generator=CCNN(num_of_stages=4, input_dim_of_first_layer=1, output_dim_of_first_layer=4,
                                  output_channels=1),
        slm_phase_generator=CCNN(num_of_stages=3, input_dim_of_first_layer=1, output_dim_of_first_layer=4,
                                 output_channels=1),
        num_of_frames=1,
        complex_model=True,
        wavelengths=[638e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    net_g = DoubleStage(
        init_phase_generator=CCNN(num_of_stages=4, input_dim_of_first_layer=1, output_dim_of_first_layer=4,
                                  output_channels=1),
        slm_phase_generator=CCNN(num_of_stages=3, input_dim_of_first_layer=1, output_dim_of_first_layer=4,
                                 output_channels=1),
        num_of_frames=1,
        complex_model=True,
        wavelengths=[520e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    net_b = DoubleStage(
        init_phase_generator=CCNN(num_of_stages=4, input_dim_of_first_layer=1, output_dim_of_first_layer=4,
                                  output_channels=1),
        slm_phase_generator=CCNN(num_of_stages=3, input_dim_of_first_layer=1, output_dim_of_first_layer=4,
                                 output_channels=1),
        num_of_frames=1,
        complex_model=True,
        wavelengths=[450e-9],
        distances=[20e-2],
        learning_rate=1e-4,
    ).to(device)
    checkpoint_r = torch.load(
        '/home/machine1/data/lightning_logs/CCNN_red_1024_1920/version_0/checkpoints/epoch=2274-step=113750.ckpt')
    net_r.load_state_dict(checkpoint_r['state_dict'])
    checkpoint_g = torch.load(
        '/home/machine1/data/lightning_logs/CCNN_green_1024_1920/version_0/checkpoints/epoch=2275-step=113800.ckpt')
    net_g.load_state_dict(checkpoint_g['state_dict'])
    checkpoint_b = torch.load(
        '/home/machine1/data/lightning_logs/CCNN_blue_1024_1920/version_0/checkpoints/epoch=660-step=33050.ckpt')
    net_b.load_state_dict(checkpoint_b['state_dict'])
    resolution = (1024, 1920)

else:
    raise ValueError('select from SwinUnet/Unet/CCNN')

transform = transforms.Compose([
    transforms.Resize(resolution),
    transforms.ToTensor(),
])

dataset = Div2k('/dev/shm/div2k/val', transform=transform, color_channel='rgb')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=35)
i = 1

with open(f'images/{model}/metric_log.txt', 'w') as file:
    with torch.no_grad():
        start_time = time.time()
        for idx, img in enumerate(dataloader):
            img = img.to(device)
            target_amp = torch.sqrt(img)
            recon_amp_r, slm_phase_r, _ = net_r(img[:, 0:1, :, :])
            recon_amp_g, slm_phase_g, _ = net_g(img[:, 1:2, :, :])
            recon_amp_b, slm_phase_b, _ = net_b(img[:, 2:3, :, :])

            recon_amp = torch.cat([recon_amp_r, recon_amp_g, recon_amp_b], dim=1)
            slm_phase = torch.cat([slm_phase_r, slm_phase_g, slm_phase_b], dim=1)

            # if i == 1:
            #     file.write(f'Hologram shape: {slm_phase.shape}\n')
            #     file.write(
            #         f'Max value in hologram:{slm_phase.max():.2f} Min value in hologram: {slm_phase.min():.2f}\n')
            recon_amp = recon_amp * (
                    torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True) / torch.sum(recon_amp * recon_amp,
                                                                                          (-2, -1), keepdim=True))
            recon_intensity = recon_amp ** 2
            slm_phase = slm_phase / (2 * torch.pi)
            # torchvision.utils.save_image(img, f'images/{model}/Target_{i}.png')
            # torchvision.utils.save_image(recon_intensity, f'images/{model}/Recon_{i}.png', normalize=True)
            # torchvision.utils.save_image(slm_phase, f'images/{model}/Hologram_{i}.png', normalize=True)
            # file.write(f'{i} th img: '
            #            f'PSNR_amp: {psnr(recon_amp, target_amp): .2f} |'
            #            f'SSIM_amp: {ssim(recon_amp, target_amp): .3f} |'
            #            f'PSNR_intensity: {psnr(recon_intensity, img): .2f} |'
            #            f'SSIM_intensity: {ssim(recon_intensity, img): .3f} |'
            #            f'PSNR_srgb: {psnr(srgb_lin2gamma(recon_intensity), srgb_lin2gamma(img)): .2f} |'
            #            f'SSIM_srgb: {ssim(srgb_lin2gamma(recon_intensity), srgb_lin2gamma(img)): .3f} |'
            #            f'Max Value of Recon: {recon_intensity.max()} |'
            #            f'Max Value of Target: {img.max()} \n')
            # i += 1
        end_time = time.time()

print('Successfully Finished!')
print(f'Execution Time: {end_time - start_time:.2f} s.')
