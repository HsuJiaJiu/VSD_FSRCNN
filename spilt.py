import torch
from torch.ao.quantization.quantize import quantize_dynamic
import torch.quantization
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os

from models import FSRCNN, FSRCNN_VSD
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr

if __name__ == '__main__':
    torch.backends.quantized.engine = 'qnnpack'

    args_scale = 4
    args_img = 'data/lenna.bmp'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN_VSD(scale_factor=4).to(device)
    #print(summary(model, (1, 25, 25)))
    #model.load_state_dict(torch.load('vsd/outputs/x4/best.pth', map_location=lambda storage, loc: storage))

    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    model.load_state_dict(torch.load(
        'Qbest.pth', map_location=lambda storage, loc: storage))

    image = pil_image.open(args_img).convert('RGB')
    resultPicture = pil_image.new('RGB', (500, 500), (0, 0, 0))

    for i in range(5):
        for j in range(5):
            box = (i*100, j*100, (i+1)*100, (j+1)*100)
            temp = image.crop(box)

            image_width = (temp.width // args_scale) * args_scale
            image_height = (temp.height // args_scale) * args_scale

            hr = temp.resize((image_width, image_height),
                             resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // args_scale, hr.height //
                           args_scale), resample=pil_image.BICUBIC)

            lr, _ = preprocess(lr, device)
            hr, ycbcr = preprocess(hr, device)

            with torch.no_grad():
                preds = model(lr).clamp(0.0, 1.0)

            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array(
                [preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output),
                             0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)

            resultPicture.paste(output, (box[0], box[1]))

    resultPicture.save(args_img.replace(
        '.', '_spilt_fsrcnn_x{}.'.format(args_scale)))
