from time import time

from argparse import ArgumentParser

import torch

from torchvision.io import decode_image
from torchvision.transforms.v2 import ToDtype, CenterCrop, Grayscale
from torchvision.utils import make_grid, save_image

from src.magiccolor.model import MagicColor

import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(description="Super-resolution upscaling script")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)

    model = MagicColor(**checkpoint["colorizer_args"])

    model.generator.add_weight_norms()

    state_dict = checkpoint["colorizer"]

    # Compensate for compiled state dict.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    model.generator.remove_parameterizations()

    model = model.to(args.device)

    model.eval()

    print("Model checkpoint loaded successfully")

    center_crop = CenterCrop((768, 768))
    rgb_to_grayscale = Grayscale(num_output_channels=1)
    to_tensor = ToDtype(torch.float32, scale=True)

    image = decode_image(args.image_path, mode="RGB")

    x = center_crop.forward(image)
    x = rgb_to_grayscale.forward(x)
    x = to_tensor.forward(x)

    x = x.unsqueeze(0).to(args.device)

    print("Colorizing ...")

    with torch.inference_mode():
        y_pred = model.colorize(x).squeeze(0)

    pair = torch.stack(
        [
            x.squeeze(0).expand(3, -1, -1),
            y_pred,
        ],
        dim=0,
    )

    grid = make_grid(pair, nrow=2)

    grid = grid.permute(1, 2, 0).to("cpu")

    plt.imshow(grid)
    plt.show()

    if "y" in input("Save image? (yes|no): ").lower():
        filename = f"out_{time()}"

        save_image(y_pred, f"{filename}.png")


if __name__ == "__main__":
    main()
