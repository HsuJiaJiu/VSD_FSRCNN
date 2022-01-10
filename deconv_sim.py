import collections
from typing import OrderedDict
import torch
from torch._C import ErrorReport
from torch.functional import Tensor
import numpy as np

from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
import PIL.Image as pil_image
from models import FSRCNN_VSD


def deconv_sim():
    x = torch.rand(1, 1, 25, 25)
    deconv = torch.nn.ConvTranspose2d(
        1, 1, kernel_size=9, stride=4, padding=4, output_padding=3, bias=False)

    a = collections.OrderedDict()
    a = deconv.state_dict()

    conv_list = list()

    for kernel_num in range(16):
        init_u = (kernel_num % 4)
        init_v = ((kernel_num // 4) % 4)

        temp = collections.OrderedDict()
        temp['weight'] = torch.zeros(1, 1, 3, 3)

        for i in range(3):
            for j in range(3):
                target_u = init_u + 4 * j
                target_v = init_v + 4 * i
                if (target_u < 9 and target_v < 9):
                    temp['weight'][0][0][2-i][2 -
                                              j] = a['weight'][0][0][target_v][target_u]

        conv_list.append(temp)

    res_spilt = torch.rand(1, 1, 100, 100)

    for kernel_num in range(16):
        init_u = (kernel_num % 4)
        init_v = ((kernel_num // 4) % 4)

        conv = torch.nn.Conv2d(1, 1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        conv.load_state_dict(conv_list[kernel_num])
        with torch.no_grad():
            temp2 = conv(x)

        for i in range(25):
            for j in range(25):
                target_u = init_u + 4 * j
                target_v = init_v + 4 * i
                if (target_u < 100 and target_v < 100):
                    res_spilt[0][0][target_v][target_u] = temp2[0][0][i][j]

    with torch.no_grad():
        res = deconv(x)

    res = torch.squeeze(res)
    res_spilt = torch.squeeze(res_spilt)
    print(res.size())
    print(res_spilt.size())

    error = 0
    for i in range(100):
        for j in range(100):
            if (res[i][j] - res_spilt[i][j]) > 0.001:
                error += 1

    print(error)

    np.savetxt('res.txt', np.squeeze(res))
    np.savetxt('res_spilt.txt', np.squeeze(res_spilt))


def layer8_deconv_sim():
    torch.backends.quantized.engine = 'qnnpack'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FSRCNN_VSD(scale_factor=4).to(device)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(torch.load(
        'Qbest.pth', map_location=lambda storage, loc: storage))

    layer8_weight = collections.OrderedDict()
    layer8_weight['weight'] = torch.zeros(56, 1, 9, 9)

    for i in range(56):
        for j in range(9):
            for k in range(9):
                layer8_weight['weight'][i][0][j][k] = torch.int_repr(
                    model.state_dict()['last_part.0.weight'][i][0][j][k]).item()

    temp = collections.OrderedDict()
    temp['weight'] = torch.zeros(16, 56, 3, 3)

    for kernel_num in range(16):
        init_u = (kernel_num % 4)
        init_v = ((kernel_num // 4) % 4)

        for i in range(56):
            for j in range(3):
                for k in range(3):
                    target_u = init_u + 4 * k
                    target_v = init_v + 4 * j
                    if (target_u < 9 and target_v < 9):
                        temp['weight'][kernel_num][i][2-j][2 -
                                                           k] = layer8_weight['weight'][i][0][target_v][target_u]

    deconv = torch.nn.ConvTranspose2d(
        56, 1, kernel_size=9, stride=4, padding=4, output_padding=3, bias=False)
    deconv.load_state_dict(layer8_weight)
    conv = torch.nn.Conv2d(56, 16, kernel_size=3,
                           stride=1, padding=1, bias=False)
    conv.load_state_dict(temp)

    fin = torch.randn(1, 56, 25, 25)
    spilt = torch.zeros(1, 1, 100, 100)

    with torch.no_grad():
        fout = deconv(fin)
        fout_temp = conv(fin)

    for kernel_num in range(16):
        init_u = (kernel_num % 4)
        init_v = ((kernel_num // 4) % 4)

        for i in range(25):
            for j in range(25):
                target_u = init_u + 4 * j
                target_v = init_v + 4 * i
                if (target_u < 100 and target_v < 100):
                    spilt[0][0][target_v][target_u] = fout_temp[0][kernel_num][i][j]

    # np.savetxt('res.txt',np.squeeze(fout))
    # np.savetxt('res_spilt.txt',np.squeeze(spilt))

    error = 0
    for i in range(100):
        for j in range(100):
            if (fout[0][0][i][j] - spilt[0][0][i][j]) > 0.001:
                error += 1

    print(error)

    torch.save(conv.state_dict(), "layer8_spilt.pth")


def layer8_spilt_out():

    with open('FeatureMap/Layer8/conv_in', 'r') as f:
        fin = torch.from_numpy(
            np.array(f.read().split(), dtype='f').reshape(1, 56, 25, 25))

    conv = torch.nn.Conv2d(56, 16, kernel_size=3,
                           stride=1, padding=1, bias=False)
    conv.load_state_dict(torch.load('layer8_spilt.pth',
                         map_location=lambda storage, loc: storage))

    with torch.no_grad():
        temp = conv(fin)

    # 210.5 = M 驗證記得改回來
    temp = (temp / 210.5).round()

    fout = torch.zeros(1, 1, 100, 100)

    #Rearrange
    for i in range(16):
        init_u = (i % 4)
        init_v = ((i // 4) % 4)
        for j in range(25):
            for k in range(25):
                target_u = init_u + 4 * k
                target_v = init_v + 4 * j
                if (target_u < 100 and target_v < 100):
                    fout[0][0][target_v][target_u] = temp[0][i][j][k]

    fout = fout.numpy().reshape(100, 100)

    ycbcr = np.load('FeatureMap/ycbcr.npy')

    output = np.array([fout, ycbcr[..., 1], ycbcr[..., 2]]
                      ).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save('test.bmp')


if __name__ == "__main__":
    layer8_spilt_out()
