from html import parser
import random

from functools import partial

from argparse import ArgumentParser

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

from torchmetrics.classification import BinaryPrecision, BinaryRecall

from data import ImageFolder
from src.magiccolor.model import MagicColor, ColorCrush, Bouncer
from loss import RelativisticBCELoss

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Generative adversarial fine-tuning script.")

    parser.add_argument("--base_checkpoint_path", type=str, required=True)
    parser.add_argument("--train_images_path", default="./dataset/train", type=str)
    parser.add_argument("--test_images_path", default="./dataset/test", type=str)
    parser.add_argument("--num_dataset_processes", default=4, type=int)
    parser.add_argument("--target_resolution", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int)
    parser.add_argument("--generator_learning_rate", default=1e-4, type=float)
    parser.add_argument("--generator_max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--critic_learning_rate", default=5e-4, type=float)
    parser.add_argument("--critic_max_gradient_norm", default=5.0, type=float)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--critic_warmup_epochs", default=2, type=int)
    parser.add_argument(
        "--color_critic_model_size",
        default="medium",
        choices=Bouncer.AVAILABLE_MODEL_SIZES,
    )
    parser.add_argument(
        "--grayscale_critic_model_size",
        default="small",
        choices=Bouncer.AVAILABLE_MODEL_SIZES,
    )
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

    if args.generator_learning_rate < 0 or args.critic_learning_rate < 0:
        raise ValueError(f"Learning rate must be a positive value.")

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

    checkpoint = torch.load(
        args.base_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

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

    colorizer_args = checkpoint["colorizer_args"]

    colorizer = MagicColor(**colorizer_args)

    colorizer.generator.add_weight_norms()

    colorizer = torch.compile(colorizer)

    colorizer.load_state_dict(checkpoint["colorizer"])

    colorizer = colorizer.to(args.device)

    crusher_args = checkpoint["crusher_args"]

    crusher = ColorCrush(**crusher_args)

    crusher.generator.add_weight_norms()

    crusher = torch.compile(crusher)

    crusher.load_state_dict(checkpoint["crusher"])

    crusher = crusher.to(args.device)

    print("Base model loaded successfully")

    color_critic_args = {
        "input_channels": 3,
        "model_size": args.color_critic_model_size,
    }

    color_critic = Bouncer.from_preconfigured(**color_critic_args)

    color_critic.add_spectral_norms()

    color_critic = color_critic.to(args.device)

    grayscale_critic_args = {
        "input_channels": 1,
        "model_size": args.grayscale_critic_model_size,
    }

    grayscale_critic = Bouncer.from_preconfigured(**grayscale_critic_args)

    grayscale_critic.add_spectral_norms()

    grayscale_critic = grayscale_critic.to(args.device)

    pixel_l1_loss = L1Loss()
    stage_1_l1_loss = L1Loss()
    bce_loss = RelativisticBCELoss()

    colorizer_optimizer = AdamW(colorizer.parameters(), lr=args.generator_learning_rate)
    crusher_optimizer = AdamW(crusher.parameters(), lr=args.generator_learning_rate)
    color_critic_optimizer = AdamW(
        color_critic.parameters(), lr=args.generator_learning_rate
    )
    grayscale_critic_optimizer = AdamW(
        grayscale_critic.parameters(), lr=args.critic_learning_rate
    )

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        colorizer.load_state_dict(checkpoint["colorizer"])
        colorizer_optimizer.load_state_dict(checkpoint["colorizer_optimizer"])

        crusher.load_state_dict(checkpoint["crusher"])
        crusher_optimizer.load_state_dict(checkpoint["crusher_optimizer"])

        color_critic.load_state_dict(checkpoint["color_critic"])
        color_critic_optimizer.load_state_dict(checkpoint["color_critic_optimizer"])

        grayscale_critic.load_state_dict(checkpoint["grayscale_critic"])
        grayscale_critic_optimizer.load_state_dict(
            checkpoint["grayscale_critic_optimizer"]
        )

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    if args.activation_checkpointing:
        colorizer.generator.enable_activation_checkpointing()
        crusher.generator.enable_activation_checkpointing()
        color_critic.detector.enable_activation_checkpointing()
        grayscale_critic.detector.enable_activation_checkpointing()

    print(
        f"Colorizer has {colorizer.generator.num_trainable_params:,} trainable parameters"
    )
    print(
        f"Crusher has {crusher.generator.num_trainable_params:,} trainable parameters"
    )
    print(
        f"Color Critic has {color_critic.num_trainable_params:,} trainable parameters"
    )
    print(
        f"Grayscale Critic has {grayscale_critic.num_trainable_params:,} trainable parameters"
    )

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    vif_metric = VisualInformationFidelity().to(args.device)

    precision_metric = BinaryPrecision().to(args.device)
    recall_metric = BinaryRecall().to(args.device)

    print("Fine-tuning ...")

    colorizer.train()
    crusher.train()
    color_critic.train()
    grayscale_critic.train()

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_colorizer_pixel_l1, total_crusher_pixel_l1 = 0.0, 0.0
        total_colorizer_stage_1_l1, total_crusher_stage_1_l1 = 0.0, 0.0
        total_colorizer_bce, total_crusher_bce = 0.0, 0.0
        total_color_critic_bce, total_grayscale_critic_bce = 0.0, 0.0
        total_colorizer_gradient_norm, total_crusher_gradient_norm = 0.0, 0.0
        total_color_critic_gradient_norm, total_grayscale_critic_gradient_norm = (
            0.0,
            0.0,
        )
        total_batches, total_steps = 0, 0

        is_warmup = epoch <= args.critic_warmup_epochs

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            y_real = torch.full((y.size(0), 1), 1.0).to(args.device)
            y_fake = torch.full((y.size(0), 1), 0.0).to(args.device)

            update_this_step = step % args.gradient_accumulation_steps == 0

            with amp_context:
                y_pred = colorizer.forward(x)

                _, _, _, _, c_pred_fake = color_critic.forward(y_pred.detach())
                _, _, _, _, c_pred_real = color_critic.forward(y)

                color_critic_bce = bce_loss.forward(
                    c_pred_real, c_pred_fake, y_real, y_fake
                )

                x_pred = crusher.forward(y_pred)

                _, _, _, _, g_pred_fake = grayscale_critic.forward(x_pred.detach())
                _, _, _, _, g_pred_real = grayscale_critic.forward(x)

                grayscale_critic_bce = bce_loss.forward(
                    g_pred_real, g_pred_fake, y_real, y_fake
                )

                combined_loss = color_critic_bce + grayscale_critic_bce
                combined_loss *= 0.5

                scaled_critic_loss = combined_loss / args.gradient_accumulation_steps

            scaled_critic_loss.backward()

            if update_this_step:
                color_critic_norm = clip_grad_norm_(
                    color_critic.parameters(), args.critic_max_gradient_norm
                )
                grayscale_critic_norm = clip_grad_norm_(
                    grayscale_critic.parameters(), args.critic_max_gradient_norm
                )

                color_critic_optimizer.step()
                grayscale_critic_optimizer.step()

                color_critic_optimizer.zero_grad()
                grayscale_critic_optimizer.zero_grad()

                total_color_critic_gradient_norm += color_critic_norm.item()
                total_grayscale_critic_gradient_norm += grayscale_critic_norm.item()

                total_steps += 1

            total_color_critic_bce += color_critic_bce.item()
            total_grayscale_critic_bce += grayscale_critic_bce.item()

            if not is_warmup:
                with amp_context:
                    colorizer_pixel_l1 = pixel_l1_loss.forward(y_pred, y)

                    c_z1_fake, _, _, _, c_pred_fake = color_critic.forward(y_pred)
                    c_z1_real, _, _, _, c_pred_real = color_critic.forward(y)

                    colorizer_stage_1_l1 = stage_1_l1_loss.forward(c_z1_fake, c_z1_real)

                    colorizer_bce = bce_loss.forward(
                        c_pred_real, c_pred_fake, y_fake, y_real
                    )

                    grayscale_critic_pixel_l1 = pixel_l1_loss.forward(x_pred, x)

                    g_z1_fake, _, _, _, g_pred_fake = grayscale_critic.forward(x_pred)
                    g_z1_real, _, _, _, g_pred_real = grayscale_critic.forward(x)

                    grayscale_critic_stage_1_l1 = stage_1_l1_loss.forward(
                        g_z1_fake, g_z1_real
                    )

                    grayscale_critic_bce = bce_loss.forward(
                        g_pred_real, g_pred_fake, y_fake, y_real
                    )

                    combined_loss = (
                        colorizer_pixel_l1 / colorizer_pixel_l1.detach()
                        + colorizer_stage_1_l1 / colorizer_stage_1_l1.detach()
                        + colorizer_bce / colorizer_bce.detach()
                        + grayscale_critic_pixel_l1 / grayscale_critic_pixel_l1.detach()
                        + grayscale_critic_stage_1_l1
                        / grayscale_critic_stage_1_l1.detach()
                        + grayscale_critic_bce / grayscale_critic_bce.detach()
                    )

                    scaled_loss = combined_loss / args.gradient_accumulation_steps

                scaled_loss.backward()

                if update_this_step:
                    colorizer_norm = clip_grad_norm_(
                        colorizer.parameters(), args.generator_max_gradient_norm
                    )
                    crusher_norm = clip_grad_norm_(
                        crusher.parameters(), args.generator_max_gradient_norm
                    )

                    colorizer_optimizer.step()
                    crusher_optimizer.step()

                    colorizer_optimizer.zero_grad()
                    crusher_optimizer.zero_grad()

                    total_colorizer_gradient_norm += colorizer_norm.item()
                    total_crusher_gradient_norm += crusher_norm.item()

                total_colorizer_pixel_l1 += colorizer_pixel_l1.item()
                total_colorizer_stage_1_l1 += colorizer_stage_1_l1.item()
                total_colorizer_bce += colorizer_bce.item()
                total_crusher_pixel_l1 += grayscale_critic_pixel_l1.item()
                total_crusher_stage_1_l1 += grayscale_critic_stage_1_l1.item()
                total_crusher_bce += grayscale_critic_bce.item()

            total_batches += 1

        average_colorizer_pixel_l1 = total_colorizer_pixel_l1 / total_batches
        average_colorizer_stage_1_l1 = total_colorizer_stage_1_l1 / total_batches
        average_colorizer_bce = total_colorizer_bce / total_batches
        average_crusher_pixel_l1 = total_crusher_pixel_l1 / total_batches
        average_crusher_stage_1_l1 = total_crusher_stage_1_l1 / total_batches
        average_crusher_bce = total_crusher_bce / total_batches
        average_color_critic_bce = total_color_critic_bce / total_batches
        average_color_critic_gradient_norm = (
            total_color_critic_gradient_norm / total_steps
        )
        average_grayscale_critic_bce = total_grayscale_critic_bce / total_batches
        average_grayscale_critic_gradient_norm = (
            total_grayscale_critic_gradient_norm / total_steps
        )

        average_colorizer_gradient_norm = total_colorizer_gradient_norm / total_steps
        average_crusher_gradient_norm = total_crusher_gradient_norm / total_steps

        logger.add_scalar("Colorizer Pixel L1", average_colorizer_pixel_l1, epoch)
        logger.add_scalar("Colorizer Stage 1 L1", average_colorizer_stage_1_l1, epoch)
        logger.add_scalar("Colorizer BCE", average_colorizer_bce, epoch)
        logger.add_scalar("Colorizer Norm", average_colorizer_gradient_norm, epoch)
        logger.add_scalar("Crusher Pixel L1", average_crusher_pixel_l1, epoch)
        logger.add_scalar("Crusher Stage 1 L1", average_crusher_stage_1_l1, epoch)
        logger.add_scalar("Crusher BCE", average_crusher_bce, epoch)
        logger.add_scalar("Crusher Norm", average_crusher_gradient_norm, epoch)
        logger.add_scalar("Color Critic BCE", average_color_critic_bce, epoch)
        logger.add_scalar(
            "Color Critic Norm", average_color_critic_gradient_norm, epoch
        )
        logger.add_scalar("Grayscale Critic BCE", average_grayscale_critic_bce, epoch)
        logger.add_scalar(
            "Grayscale Critic Norm", average_grayscale_critic_gradient_norm, epoch
        )

        print(
            f"Epoch {epoch}:",
            f"Colorizer Pixel L1: {average_colorizer_pixel_l1:.5},",
            f"Colorizer Stage 1 L1: {average_colorizer_stage_1_l1:.5},",
            f"Colorizer BCE: {average_colorizer_bce:.5},",
            f"Colorizer Norm: {average_colorizer_gradient_norm:.4},",
            f"Crusher Pixel L1: {average_crusher_pixel_l1:.5},",
            f"Crusher Stage 1 L1: {average_crusher_stage_1_l1:.5},",
            f"Crusher BCE: {average_crusher_bce:.5},",
            f"Crusher Norm: {average_crusher_gradient_norm:.4}",
            f"Color Critic BCE: {average_color_critic_bce:.5},",
            f"Color Critic Norm: {average_color_critic_gradient_norm:.4},",
            f"Grayscale Critic BCE: {average_grayscale_critic_bce:.5},",
            f"Grayscale Critic Norm: {average_grayscale_critic_gradient_norm:.4}",
        )

        if epoch % args.eval_interval == 0:
            colorizer.eval()
            color_critic.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                y_real = torch.full((y.size(0), 1), 1.0).to(args.device)
                y_fake = torch.full((y.size(0), 1), 0.0).to(args.device)

                y_pred = colorizer.colorize(x)

                c_pred_real = color_critic.predict(y)
                c_pred_fake = color_critic.predict(y_pred)

                c_pred_real -= c_pred_fake.mean()
                c_pred_fake -= c_pred_real.mean()

                c_pred = torch.cat((c_pred_real, c_pred_fake), dim=0)
                labels = torch.cat((y_real, y_fake), dim=0)

                psnr_metric.update(y_pred, y)
                ssim_metric.update(y_pred, y)
                vif_metric.update(y_pred, y)

                precision_metric.update(c_pred, labels)
                recall_metric.update(c_pred, labels)

            psnr = psnr_metric.compute()
            ssim = ssim_metric.compute()
            vif = vif_metric.compute()

            precision = precision_metric.compute()
            recall = recall_metric.compute()

            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

            logger.add_scalar("PSNR", psnr, epoch)
            logger.add_scalar("SSIM", ssim, epoch)
            logger.add_scalar("VIF", vif, epoch)
            logger.add_scalar("F1 Score", f1_score, epoch)
            logger.add_scalar("Precision", precision, epoch)
            logger.add_scalar("Recall", recall, epoch)

            print(
                f"PSNR: {psnr:.5},",
                f"SSIM: {ssim:.5},",
                f"VIF: {vif:.5},",
                f"F1 Score: {f1_score:.5},",
                f"Precision: {precision:.5},",
                f"Recall: {recall:.5}",
            )

            psnr_metric.reset()
            ssim_metric.reset()
            vif_metric.reset()

            precision_metric.reset()
            recall_metric.reset()

            colorizer.train()
            color_critic.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "colorizer_args": colorizer_args,
                "colorizer": colorizer.state_dict(),
                "colorizer_optimizer": colorizer_optimizer.state_dict(),
                "crusher_args": crusher_args,
                "crusher": crusher.state_dict(),
                "crusher_optimizer": crusher_optimizer.state_dict(),
                "color_critic_args": color_critic_args,
                "color_critic": color_critic.state_dict(),
                "color_critic_optimizer": color_critic_optimizer.state_dict(),
                "grayscale_critic_args": grayscale_critic_args,
                "grayscale_critic": grayscale_critic.state_dict(),
                "grayscale_critic_optimizer": grayscale_critic_optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
