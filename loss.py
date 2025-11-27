import torch

from torch import Tensor

from torch.nn import Module, MSELoss, BCEWithLogitsLoss


class RelativisticBCELoss(Module):
    """
    Relativistic average BCE with logits loss for generative adversarial network training.
    """

    def __init__(self):
        super().__init__()

        self.bce = BCEWithLogitsLoss()

    def forward(
        self,
        y_pred_real: Tensor,
        y_pred_fake: Tensor,
        y_real: Tensor,
        y_fake: Tensor,
    ) -> Tensor:
        y_pred_real_hat = y_pred_real - y_pred_fake.mean()
        y_pred_fake_hat = y_pred_fake - y_pred_real.mean()

        y_pred = torch.cat((y_pred_real_hat, y_pred_fake_hat))
        y = torch.cat((y_real, y_fake))

        loss = self.bce.forward(y_pred, y)

        return loss
