from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configurations import DEVICE, BATCH_SIZE, DTYPE, DIM
from configurations import conf
from dataset import MattingDataset
from depth import Midas
from loss import overall_loss
from mapping import fg_bg_consistency_map
from metrics import mean_absolute_error
from unet import UNet
from utils import save_checkpoint, save_predictions, add_depth_to_input

LEARNING_RATE = conf['hparam']['lr']
EPOCH = conf['hparam']['epoch']

DATA_DIR = Path(conf["dataset"]['data_dir'])
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
BG_DIR = conf["dataset"]['bg_dir']

MODEL_CHECKPOINT_DIR = conf['checkpoints']['model_checkpoint_dir']
IMAGE_CHECKPOINT_DIR = conf['checkpoints']['image_checkpoint_dir']
MODEL_PATH_TO_INITIALIZE = Path(MODEL_CHECKPOINT_DIR) / conf['checkpoints']['model_path_to_initialize']

# MiDaS model
# model_type = "DPT_Large"
# model_type = "DPT_Hybrid"
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device(DEVICE)
midas.to(device)
midas.eval()
midas = Midas(model=midas, output_dim=DIM)

# data loaders
train_dataset = MattingDataset(data_dir=TRAIN_DIR,
                               bg_dir=BG_DIR,
                               consistency_map=fg_bg_consistency_map,
                               with_augmentation=True)
val_dataset = MattingDataset(data_dir=VAL_DIR,
                             bg_dir=BG_DIR,
                             consistency_map=fg_bg_consistency_map,
                             with_augmentation=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)


# training
def train(loader, model, optimizer, scaler):
    """One epoch of training
    """
    loop = tqdm(loader)
    loop.set_description("Train: ")

    for batch_idx, (X, y, X_for_depth) in enumerate(loop):
        y_gt = y.to(DEVICE)
        X_new = add_depth_to_input(x=X, x_for_depth=X_for_depth,
                                   device=DEVICE,
                                   dtype=DTYPE,
                                   output_dim=DIM,
                                   model=midas)

        # forward
        with torch.cuda.amp.autocast():
            y_pr = model(X_new)
            loss = overall_loss(y_pr, y_gt)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = UNet(in_ch=4).to(DEVICE)
    # checkpoint = torch.load(MODEL_PATH_TO_INITIALIZE)
    # load_checkpoint(checkpoint, unet)
    # model = UNetRefiner(unet).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    model_name = model.__class__.__name__

    for epoch in range(EPOCH):
        train(train_loader, model, optimizer, scaler)

        # error
        mae = mean_absolute_error(val_loader, model, model_depth=midas, device=DEVICE)
        print("Validation MAE: ", mae.item())

        # model checkpoint
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, MODEL_CHECKPOINT_DIR, model_name)

        # save predictions
        save_predictions(val_loader, DIM, DTYPE, midas, model, IMAGE_CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
