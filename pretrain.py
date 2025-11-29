import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.utils.data import DataLoader
from torch.nn import L1Loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms.v2 import (
    Compose,
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
)

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)

from data import ImageFolder
from src.magiccolor.model import MagicColor, ColorCrush

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Pretraining script.")

    parser.add_argument("--train_images_path", default="./dataset/train", type=str)
    parser.add_argument("--test_images_path", default="./dataset/test", type=str)
    parser.add_argument("--num_dataset_processes", default=4, type=int)
    parser.add_argument("--target_resolution", default=256, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--max_gradient_norm", default=2.0, type=float)
    parser.add_argument("--colorizer_primary_channels", default=32, type=int)
    parser.add_argument("--colorizer_primary_layers", default=4, type=int)
    parser.add_argument("--colorizer_secondary_channels", default=64, type=int)
    parser.add_argument("--colorizer_secondary_layers", default=4, type=int)
    parser.add_argument("--colorizer_tertiary_channels", default=128, type=int)
    parser.add_argument("--colorizer_tertiary_layers", default=6, type=int)
    parser.add_argument("--colorizer_quaternary_channels", default=256, type=int)
    parser.add_argument("--colorizer_quaternary_layers", default=6, type=int)
    parser.add_argument("--colorizer_hidden_ratio", default=2, type=int)
    parser.add_argument("--crusher_primary_channels", default=16, type=int)
    parser.add_argument("--crusher_primary_layers", default=2, type=int)
    parser.add_argument("--crusher_secondary_channels", default=32, type=int)
    parser.add_argument("--crusher_secondary_layers", default=2, type=int)
    parser.add_argument("--crusher_tertiary_channels", default=64, type=int)
    parser.add_argument("--crusher_tertiary_layers", default=4, type=int)
    parser.add_argument("--crusher_quaternary_channels", default=128, type=int)
    parser.add_argument("--crusher_quaternary_layers", default=4, type=int)
    parser.add_argument("--crusher_hidden_ratio", default=2, type=int)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval_interval", default=2, type=int)
    parser.add_argument("--checkpoint_interval", default=2, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if "mps" in args.device and not mps_is_available():
        raise RuntimeError("MPS is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    new_dataset = partial(ImageFolder, target_resolution=args.target_resolution)

    training = new_dataset(
        args.train_images_path,
        pre_transform=Compose(
            [
                RandomCrop(args.target_resolution),
                RandomHorizontalFlip(),
            ]
        ),
    )

    testing = new_dataset(
        args.test_images_path,
        pre_transform=CenterCrop(args.target_resolution),
    )

    new_dataloader = partial(
        DataLoader,
        batch_size=args.batch_size,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    train_loader = new_dataloader(training, shuffle=True)
    test_loader = new_dataloader(testing)

    colorizer_args = {
        "primary_channels": args.colorizer_primary_channels,
        "primary_layers": args.colorizer_primary_layers,
        "secondary_channels": args.colorizer_secondary_channels,
        "secondary_layers": args.colorizer_secondary_layers,
        "tertiary_channels": args.colorizer_tertiary_channels,
        "tertiary_layers": args.colorizer_tertiary_layers,
        "quaternary_channels": args.colorizer_quaternary_channels,
        "quaternary_layers": args.colorizer_quaternary_layers,
        "hidden_ratio": args.colorizer_hidden_ratio,
    }

    colorizer = MagicColor(**colorizer_args)

    colorizer.generator.add_weight_norms()

    colorizer = colorizer.to(args.device)

    crusher_args = {
        "primary_channels": args.crusher_primary_channels,
        "primary_layers": args.crusher_primary_layers,
        "secondary_channels": args.crusher_secondary_channels,
        "secondary_layers": args.crusher_secondary_layers,
        "tertiary_channels": args.crusher_tertiary_channels,
        "tertiary_layers": args.crusher_tertiary_layers,
        "quaternary_channels": args.crusher_quaternary_channels,
        "quaternary_layers": args.crusher_quaternary_layers,
        "hidden_ratio": args.crusher_hidden_ratio,
    }

    crusher = ColorCrush(**crusher_args)

    crusher.generator.add_weight_norms()

    crusher = crusher.to(args.device)

    l1_loss_function = L1Loss()

    print("Compiling models")

    colorizer = torch.compile(colorizer)
    crusher = torch.compile(crusher)

    print(
        f"Colorizer has {colorizer.generator.num_trainable_params:,} trainable parameters"
    )
    print(
        f"Crusher has {crusher.generator.num_trainable_params:,} trainable parameters"
    )

    colorizer_optimizer = AdamW(colorizer.parameters(), lr=args.learning_rate)
    crusher_optimizer = AdamW(crusher.parameters(), lr=args.learning_rate)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    vif_metric = VisualInformationFidelity().to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        colorizer.load_state_dict(checkpoint["colorizer"])
        colorizer_optimizer.load_state_dict(checkpoint["colorizer_optimizer"])

        crusher.load_state_dict(checkpoint["crusher"])
        crusher_optimizer.load_state_dict(checkpoint["crusher_optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    if args.activation_checkpointing:
        colorizer.generator.enable_activation_checkpointing()
        crusher.generator.enable_activation_checkpointing()

    print("Training ...")
    colorizer.train()
    crusher.train()

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_colorizer_l1_loss, total_crusher_l1_loss = 0.0, 0.0
        total_batches, total_steps = 0, 0
        total_colorizer_gradient_norm, total_crusher_gradient_norm = 0.0, 0.0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                y_pred = colorizer.forward(x)

                colorizer_l1_loss = l1_loss_function(y_pred, y)

                x_pred = crusher.forward(y_pred)

                crusher_l1_loss = l1_loss_function(x_pred, x)

                combined_loss = colorizer_l1_loss + crusher_l1_loss

                scaled_loss = combined_loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                colorizer_norm = clip_grad_norm_(
                    colorizer.parameters(), args.max_gradient_norm
                )
                crusher_norm = clip_grad_norm_(
                    crusher.parameters(), args.max_gradient_norm
                )

                colorizer_optimizer.step()
                crusher_optimizer.step()

                colorizer_optimizer.zero_grad()
                crusher_optimizer.zero_grad()

                total_colorizer_gradient_norm += colorizer_norm.item()
                total_crusher_gradient_norm += crusher_norm.item()

                total_steps += 1

            total_colorizer_l1_loss += colorizer_l1_loss.item()
            total_crusher_l1_loss += crusher_l1_loss.item()

            total_batches += 1

        average_colorizer_l1_loss = total_colorizer_l1_loss / total_batches
        average_crusher_l1_loss = total_crusher_l1_loss / total_batches
        average_colorizer_gradient_norm = total_colorizer_gradient_norm / total_steps
        average_crusher_gradient_norm = total_crusher_gradient_norm / total_steps

        logger.add_scalar("Colorizer Pixel L1", average_colorizer_l1_loss, epoch)
        logger.add_scalar("Crusher Pixel L1", average_crusher_l1_loss, epoch)
        logger.add_scalar(
            "Colorizer Gradient Norm", average_colorizer_gradient_norm, epoch
        )
        logger.add_scalar("Crusher Gradient Norm", average_crusher_gradient_norm, epoch)

        print(
            f"Epoch {epoch}:",
            f"Colorizer Pixel L1: {average_colorizer_l1_loss:.4},",
            f"Crusher Pixel L1: {average_crusher_l1_loss:.4},",
            f"Colorizer Gradient Norm: {average_colorizer_gradient_norm:.4},",
            f"Crusher Gradient Norm: {average_crusher_gradient_norm:.4}",
        )

        if epoch % args.eval_interval == 0:
            colorizer.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                y_pred = colorizer.colorize(x)

                psnr_metric.update(y_pred, y)
                ssim_metric.update(y_pred, y)
                vif_metric.update(y_pred, y)

            psnr = psnr_metric.compute()
            ssim = ssim_metric.compute()
            vif = vif_metric.compute()

            logger.add_scalar("PSNR", psnr, epoch)
            logger.add_scalar("SSIM", ssim, epoch)
            logger.add_scalar("VIF", vif, epoch)

            print(
                f"PSNR: {psnr:.4},",
                f"SSIM: {ssim:.4},",
                f"VIF: {vif:.4}",
            )

            psnr_metric.reset()
            ssim_metric.reset()
            vif_metric.reset()

            colorizer.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "colorizer_args": colorizer_args,
                "colorizer": colorizer.state_dict(),
                "colorizer_optimizer": colorizer_optimizer.state_dict(),
                "crusher_args": crusher_args,
                "crusher": crusher.state_dict(),
                "crusher_optimizer": crusher_optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
