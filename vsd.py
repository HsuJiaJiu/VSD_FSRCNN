import torch
import numpy as np
import PIL.Image as pil_image
import os

from models import FSRCNN, FSRCNN_VSD
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
from torchsummary import summary

if __name__ == '__main__':
    torch.backends.quantized.engine = 'qnnpack'

    args_scale = 4
    args_img = 'data/lenna.bmp'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN_VSD(scale_factor=4).to(device)
    print(summary(model, (1, 25, 25)))

    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    model.load_state_dict(torch.load(
        'Qbest.pth', map_location=lambda storage, loc: storage))

    image = pil_image.open(args_img).convert('RGB')

    image_width = (image.width // args_scale) * args_scale
    image_height = (image.height // args_scale) * args_scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args_scale, hr.height //
                   args_scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args_scale, lr.height *
                        args_scale), resample=pil_image.BICUBIC)
    bicubic.save(args_img.replace('.', '_bicubic_x{}.'.format(args_scale)))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)

    bicubic, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    print('HR PSNR: {:.2f}'.format(psnr))
    psnr = calc_psnr(hr, bicubic)
    print('Bic PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]
                      ).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args_img.replace('.', '_fsrcnn_x{}.'.format(args_scale)))
