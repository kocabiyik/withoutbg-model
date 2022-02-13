import torch
from tqdm import tqdm

from configurations import DEVICE, DIM, DTYPE, BATCH_SIZE
from utils import add_depth_to_input


def mean_absolute_error(loader, model, model_depth, device):
    sum_absolute_diff = 0
    num_pixels = DIM * DIM
    model.eval()

    with torch.no_grad():
        loop = tqdm(loader)
        loop.set_description("Evaluation: ")
        for X, y, X_for_depth in loop:
            X_new = add_depth_to_input(x=X, x_for_depth=X_for_depth,
                                       device=DEVICE,
                                       dtype=DTYPE,
                                       output_dim=DIM,
                                       batch_size=BATCH_SIZE,
                                       model=model_depth)
            X_new = X_new.to(device)
            y = y.to(device)
            alpha_gt = y[:, 0:1, :, :]
            alpha_pr = model(X_new)

            sum_absolute_diff += torch.sum(torch.abs(alpha_pr - alpha_gt))
            num_pixels += torch.numel(alpha_pr)

            mae = sum_absolute_diff / num_pixels
            loop.set_postfix(mae_val=mae.item())

    model.train()
    return mae
