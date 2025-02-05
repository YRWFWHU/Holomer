from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torch
import cv2

from algorithms.SGD import SGD


class SaveHologramAfterTrain(Callback):
    """
    Used by SGD, save hologram in PNG format after optimization
    """
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def on_train_end(self, trainer: pl.Trainer, model: SGD) -> None:
        init_phase = model.phase.detach()   # range -10 ~ 10
        phase_only_hologram = torch.atan2(torch.sin(init_phase), torch.cos(init_phase))
        phase_only_hologram = phase_only_hologram.squeeze(0).permute(1, 2, 0).numpy()
        phase_only_hologram += torch.pi     # range 0 ~ 2pi
        print(phase_only_hologram.max(), phase_only_hologram.min())
        cv2.imwrite('phase_only_hologram.png', phase_only_hologram / phase_only_hologram.max() * 255.0)
        print(f'Phase only hologram shape: {phase_only_hologram.shape}')
