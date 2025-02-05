from diffusers import StableDiffusionXLImg2ImgPipeline, UNet2DModel
from diffusers.utils import load_image
import torch


class SDWithProjector(torch.nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.diffuser = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "/mnt/mnt2/wyr/Holomer/local_model", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        self.projector = UNet2DModel(
            sample_size=resolution,  # the target ima
            # ge resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, prompt):
        return self.diffuser(prompt=prompt, image=x).images


if __name__ == "__main__":
    url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
    init_image = load_image(url).convert("RGB")
    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt, image=init_image).images
    model = SDWithProjector(resolution=256).to('cuda')
    output = model(x, prompt)
    print(output.shape)
