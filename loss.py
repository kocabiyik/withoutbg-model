# Compositial Loss mentioned in Deep Image Matting Paper
# https://arxiv.org/abs/1703.03872 

import torch

from configurations import EPSILON, DIM


def overall_loss(y_pr, y_gt, weight_cl=0.5):
    alpha_pr = y_pr

    # y_gt:  GT(1), BG(3), FG(3), IMG(3)
    alpha_gt = y_gt[:, 0:1, :, :]
    bg = y_gt[:, 1:4, :, :]
    fg = y_gt[:, 4:7, :, :]
    img_gt = y_gt[:, 7:10, :, :]

    epsilon_sqr = EPSILON ** 2
    num_pixels = DIM * DIM

    # alpha loss
    alpha_diff = alpha_pr - alpha_gt
    alpha_diff = torch.sqrt(torch.pow(alpha_diff, 2) + epsilon_sqr)
    alpha_loss = torch.sum(alpha_diff) / (num_pixels + EPSILON)

    # compositional loss
    img_composited = fg * alpha_pr + bg * (1 - alpha_pr)
    compositional_diff = img_gt - img_composited
    compositional_diff = torch.sqrt(torch.pow(compositional_diff, 2) + epsilon_sqr)
    compositional_loss = torch.sum(compositional_diff) / (num_pixels + EPSILON)

    return weight_cl * compositional_loss + (1 - weight_cl) * alpha_loss
