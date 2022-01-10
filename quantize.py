import os
import numpy as np
import PIL.Image as pil_image
import torch
from models import FSRCNN, FSRCNN_VSD

from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
from datasets import EvalDataset
from torch.utils.data.dataloader import DataLoader
import torch.quantization


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


if __name__ == '__main__':
    torch.backends.quantized.engine = 'qnnpack'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FSRCNN_VSD(scale_factor=4).to(device)
    model.load_state_dict(torch.load(
        'vsd/outputs/x4/best.pth', map_location=lambda storage, loc: storage))
    # fuse model
    torch.quantization.fuse_modules(
        model, [["first_part.0", "first_part.1"]], inplace=True)
    torch.quantization.fuse_modules(
        model, [["mid_part.0", "mid_part.1"]], inplace=True)
    torch.quantization.fuse_modules(
        model, [["mid_part.2", "mid_part.3"]], inplace=True)
    torch.quantization.fuse_modules(
        model, [["mid_part.4", "mid_part.5"]], inplace=True)
    torch.quantization.fuse_modules(
        model, [["mid_part.6", "mid_part.7"]], inplace=True)
    torch.quantization.fuse_modules(
        model, [["mid_part.8", "mid_part.9"]], inplace=True)
    torch.quantization.fuse_modules(
        model, [["mid_part.10", "mid_part.11"]], inplace=True)
    # fuse model end
    model.eval()

    #model.qconfig = torch.quantization.default_qconfig
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    # print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)
    print_size_of_model(model)

    eval_dataset = EvalDataset("vsd/Set5_x4.h5")
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        psnr = calc_psnr(preds, labels)
        print('PSNR: {:.2f}'.format(psnr))

    torch.quantization.convert(model, inplace=True)
    print_size_of_model(model)
    print(model)

    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        psnr = calc_psnr(preds, labels)
        print('PSNR: {:.2f}'.format(psnr))

    torch.save(model, 'Qmodel.pth')
    torch.save(model.state_dict(), 'Qbest.pth')
