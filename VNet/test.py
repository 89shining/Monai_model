# ================= test.py =================
import argparse
import glob
import os

import SimpleITK as sitk
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import VNet
from monai.transforms import *

def main():
    device = torch.device("cuda")

    model = VNet(3,1,1).to(device)
    model.load_state_dict(torch.load("best.pth"))
    model.eval()

    images = sorted(glob.glob("imagesTs/*_0000.nii.gz"))

    tf = Compose([
        LoadImaged(["image"]),
        EnsureChannelFirstd(["image"]),
        ScaleIntensityRanged("image",-1000,1000,0,1,True),
        EnsureTyped(["image"])
    ])

    loader = DataLoader(Dataset([{"image":i} for i in images], tf), batch_size=1)

    for i,b in enumerate(loader):
        x = b["image"].to(device)

        pred = sliding_window_inference(
            x,(128,128,96),1,model,
            sw_device=device,
            device=torch.device("cpu")
        )

        pred = (torch.sigmoid(pred)>0.5).float()[0,0].numpy()

        img = sitk.ReadImage(images[i])
        out = sitk.GetImageFromArray(pred)
        out.CopyInformation(img)

        sitk.WriteImage(out, f"pred_{i}.nii.gz")

if __name__ == "__main__":
    main()