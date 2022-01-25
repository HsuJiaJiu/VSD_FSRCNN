import enum
from numpy.core.fromnumeric import partition
import torch
from torch.ao.quantization.quantize import quantize_dynamic
from torch.autograd.grad_mode import no_grad
import torch.quantization
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os

from models import FSRCNN, FSRCNN_VSD
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
from torchsummary import summary


def hook(module, fea_in, fea_out):
    #print("hooker working")
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None


if __name__ == '__main__':
    torch.backends.quantized.engine = 'qnnpack'

    args_scale = 4
    args_img = 'data/lenna.bmp'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN_VSD(scale_factor=4).to(device)

    print(summary(model,(1,25,25)))

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

    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    model.load_state_dict(torch.load(
        './vsd/Qbest.pth', map_location=lambda storage, loc: storage))

    box1 = (250, 250, 350, 350)
    image = pil_image.open(args_img).convert('RGB').crop(box1)

    image_width = (image.width // args_scale) * args_scale
    image_height = (image.height // args_scale) * args_scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args_scale, hr.height //
                   args_scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args_scale, lr.height *
                        args_scale), resample=pil_image.BICUBIC)

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    bicubic, ycbcr = preprocess(bicubic, device)

    module_name = []
    features_in_hook = []
    features_out_hook = []
    for child in model.first_part.children():
        if not isinstance(child, torch.nn.modules.Identity):
            child.register_forward_hook(hook=hook)

    for child in model.mid_part.children():
        if not isinstance(child, torch.nn.modules.Identity):
            child.register_forward_hook(hook=hook)

    for child in model.last_part.children():
        if not isinstance(child, torch.nn.modules.Identity):
            child.register_forward_hook(hook=hook)

    
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
    output.save('test.bmp')

    # * Weight extraction
    weightIndex = ['first_part.0.weight','mid_part.0.weight','mid_part.2.weight',
    'mid_part.4.weight','mid_part.6.weight','mid_part.8.weight','mid_part.10.weight']

    for i,j in enumerate(weightIndex ,start=1):
        temp = torch.int_repr(model.state_dict()[j]).numpy().flatten()
        with open('weight/weight_layer{}'.format(i), 'w') as f:
            np.savetxt(f,temp,fmt="%d",delimiter='\n')


    # * quantize M calculate
    s1 = list()
    s2 = list()
    s3 = list()
    M = list()

    s1.append(model.state_dict()['quant.scale'].item())
    s1.append(model.state_dict()['first_part.0.scale'].item())
    s1.append(model.state_dict()['mid_part.0.scale'].item())
    s1.append(model.state_dict()['mid_part.2.scale'].item())
    s1.append(model.state_dict()['mid_part.4.scale'].item())
    s1.append(model.state_dict()['mid_part.6.scale'].item())
    s1.append(model.state_dict()['mid_part.8.scale'].item())
    s1.append(model.state_dict()['mid_part.10.scale'].item())

    s2.append(model.state_dict()['first_part.0.weight'].q_scale())
    s2.append(model.state_dict()['mid_part.0.weight'].q_scale())
    s2.append(model.state_dict()['mid_part.2.weight'].q_scale())
    s2.append(model.state_dict()['mid_part.4.weight'].q_scale())
    s2.append(model.state_dict()['mid_part.6.weight'].q_scale())
    s2.append(model.state_dict()['mid_part.8.weight'].q_scale())
    s2.append(model.state_dict()['mid_part.10.weight'].q_scale())
    s2.append(model.state_dict()['last_part.0.weight'].q_scale())

    s3.append(model.state_dict()['first_part.0.scale'].item())
    s3.append(model.state_dict()['mid_part.0.scale'].item())
    s3.append(model.state_dict()['mid_part.2.scale'].item())
    s3.append(model.state_dict()['mid_part.4.scale'].item())
    s3.append(model.state_dict()['mid_part.6.scale'].item())
    s3.append(model.state_dict()['mid_part.8.scale'].item())
    s3.append(model.state_dict()['mid_part.10.scale'].item())
    s3.append(model.state_dict()['last_part.0.scale'].item())

    for i, j, k in zip(s1, s2, s3):
        M.append(i*j/k)


    print("="*15)
    for i, element in enumerate(M, start=1):
        m0 = element
        shift = 0
        for _ in range(30):
            #if(m0 * 2 > 1000):
            if(m0 * 2 > 65535):
                break
            m0 *= 2
            shift += 1
        print("Layer" + str(i) + " Quantization parameters")
        print("M = " + str(element))
        print("M0 = " + str(round(m0)))
        print("shift = " + str(shift))
        print("="*15)
    '''

    for w, x, y, z in zip(range(1, 9), module_name, features_in_hook, features_out_hook):
        filepath = 'FeatureMap/Layer{}/'.format(w)
        #os.makedirs(filepath)
        with open(filepath + 'conv_in', 'w+') as f:
            for i in torch.int_repr(torch.squeeze(y[0])):
                np.savetxt(f, i.numpy().tolist(), fmt="%d", delimiter='\n')
        with open(filepath + 'conv_out', 'w+') as f:
            for i in torch.int_repr(torch.squeeze(z[0])):
                np.savetxt(f, i.numpy().tolist(), fmt="%d", delimiter='\n')
        np.save('FeatureMap/ycbcr', ycbcr)

    for w, x, y, z in zip(range(1, 9), module_name, features_in_hook, features_out_hook):
        filepath = 'FeatureMap_hex/Layer{}/'.format(w)
        #os.makedirs(filepath)
        with open(filepath + 'conv_in', 'w+') as f:
            for i in torch.int_repr(torch.squeeze(y[0])):
                np.savetxt(f, i.numpy().tolist(), fmt="%x", delimiter='\n')
        with open(filepath + 'conv_out', 'w+') as f:
            for i in torch.int_repr(torch.squeeze(z[0])):
                np.savetxt(f, i.numpy().tolist(), fmt="%x", delimiter='\n')
        np.save('FeatureMap_hex/ycbcr', ycbcr)
    

    vsd_image = torch.int_repr(features_in_hook[0][0]).numpy().flatten()
    vsd_output = [0] * 1024

    tar_i = 66
    tar_j = 0
    for i, element in enumerate(vsd_image, start=1):
        vsd_output[tar_i + tar_j] = element
        tar_j += 1
        if tar_j == 25:
            tar_i += 32
            tar_j = 0

    with open('vsd_image','w+') as f:
        np.savetxt(f,vsd_output,fmt="%x",delimiter='\n')
    '''
        
    