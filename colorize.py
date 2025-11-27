from time import time

from argparse import ArgumentParser

import torch

from torchvision.io import decode_image
from torchvision.transforms.v2 import ToDtype
from torchvision.utils import make_grid, save_image
from torchvision.transforms.v2.functional import center_crop

from kornia.color import rgb_to_lab, lab_to_rgb

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

    model = MagicColor(**checkpoint["model_args"])

    model.add_weight_norms()

    state_dict = checkpoint["model"]

    # Compensate for compiled state dict.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    model.remove_parameterizations()

    model = model.to(args.device)

    model.eval()

    print("Model checkpoint loaded successfully")

    image_to_tensor = ToDtype(torch.float32, scale=True)

    rgb_image = decode_image(args.image_path, mode="RGB")

    rgb_image = center_crop(rgb_image, 1024)

    rgb_image = image_to_tensor(rgb_image)

    lab_image = rgb_to_lab(rgb_image)

    x = lab_image[0:1, :, :]

    x = x.unsqueeze(0).to(args.device)

    print("Colorizing ...")

    with torch.inference_mode():
        y_pred = model.colorize(x).squeeze(0)

    y_pred = lab_to_rgb(y_pred)

    if "y" in input("Save image? (yes|no): ").lower():
        filename = f"out_{time()}"

        save_image(y_pred, f"{filename}.png")


if __name__ == "__main__":
    main()
