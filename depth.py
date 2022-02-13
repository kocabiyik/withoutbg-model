# Using inverse depth of an image as additional input
# MiDaS by Intel ISL
# https://pytorch.org/hub/intelisl_midas_v2/

import torch


class Midas:
    def __init__(self, model, output_dim):
        self.output_dim = output_dim
        self.midas = model

    def predict_depth(self, input_batch):
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=self.output_dim,
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        pred = prediction.cpu().numpy()
        pred_normalized = pred * (1 / pred.max())
        return pred_normalized
