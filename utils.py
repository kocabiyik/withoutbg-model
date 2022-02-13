import time
from pathlib import Path

import torch
import torchvision

from configurations import DEVICE


def save_checkpoint(state, dir, model_name, file_name=None):
    if file_name is None:
        file_name = Path(dir) / f"{model_name}_{time.strftime('%Y%m%d%H%M%S')}.pth.tar"

    print("Saving checkpoint: ", file_name)
    torch.save(state, file_name)


def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def add_depth_to_input(x, x_for_depth, device, dtype, output_dim, model):
    X_for_depth = x_for_depth.to(device)
    depth = model.predict_depth(X_for_depth)
    n_batch = X_for_depth.shape[0]
    X_new = torch.zeros((n_batch, 4, output_dim, output_dim))
    X_new[:, 0:3, :, :] = x
    X_new[:, 3, :, :] = torch.tensor(depth)
    X_new = X_new.to(dtype).to(device)
    return X_new


def save_predictions(loader, DIM, DTYPE, midas, model, folder):
    model.eval()
    for idx, (X, y, X_for_depth) in enumerate(loader):
        X_new = add_depth_to_input(x=X, x_for_depth=X_for_depth,
                                   device=DEVICE,
                                   dtype=DTYPE,
                                   output_dim=DIM,
                                   model=midas)
        with torch.no_grad():
            preds = model(X_new)

        torchvision.utils.save_image(X_new[:, 0:3, :, :], f"{folder}/{idx}_0_img.png")
        torchvision.utils.save_image(X_new, f"{folder}/{idx}_1_depth.png")
        torchvision.utils.save_image(X_new[:, 3:4, :, :], f"{folder}/{idx}_2_depth_channel.png")
        torchvision.utils.save_image(preds, f"{folder}/{idx}_3_pred.png")
    model.train()
