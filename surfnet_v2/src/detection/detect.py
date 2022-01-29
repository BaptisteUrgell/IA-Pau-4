import torch
from time import time
import matplotlib.pyplot as plt
import numpy as np 
import cv2
def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def detect(preprocessed_frames, threshold, model):
    out_mod = model(preprocessed_frames)
    print(out_mod[-1]["hm"].shape)

    batch_result = torch.sigmoid(out_mod[-1]["hm"])
    print(np.squeeze(np.squeeze(batch_result.numpy(), axis=0), axis=0))
    batch_peaks = nms(batch_result).gt(threshold).squeeze(dim=1)
    detections = [torch.nonzero(peaks).cpu().numpy()[:,::-1] for peaks in batch_peaks]
    return detections
