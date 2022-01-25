from ctypes import resize
import torch
from models import FSRCNN_VSD
import PIL.Image as pil_image
import numpy as np
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr

args_img = 'Final_Img/test.bmp'
args_scale = 4

def image_pre():
    image = pil_image.open(args_img).convert('RGB').resize((500,500))

    hr = image.resize((500, 500), resample=pil_image.BICUBIC)
    hr.save('Final_Img/hr.png')

    lr = hr.resize((hr.width // args_scale, hr.height // args_scale), resample=pil_image.BICUBIC)
    lr.save('Final_Img/lr.png')
    
    bicubic = lr.resize((lr.width * args_scale, lr.height *args_scale), resample=pil_image.BICUBIC)
    bicubic.save('Final_Img/bicubic.png')

    _ , ycbcr = preprocess(bicubic, 'cpu')
    np.save('Final_Img/ycbcr', ycbcr)

    hexfile = list()
    for _ in range(25):
        hexfile.append([0] * 1024)


    for i in range(5):
        for j in range(5):
            box = (i*25, j*25, (i+1)*25, (j+1)*25)
            temp = lr.crop(box)

            temp, _ = preprocess(temp, 'cpu')
            temp = (temp / 0.003593767760321498).round().flatten().tolist()

            tar_i = 66
            tar_j = 0
            for element in temp:
                hexfile[(i * 5) + j][tar_i + tar_j] = int(element)
                tar_j += 1
                if tar_j == 25:
                    tar_i += 32
                    tar_j = 0


    for i,element in enumerate(hexfile ,start=1):
        print(element)
        with open('Final_Img/InputHex/inputhex{}'.format(i),'w+') as f:
            np.savetxt(f,element,fmt="%x",delimiter='\n')
    
    '''
    flat_hexfile = np.array([item for sublist in hexfile for item in sublist],dtype='int')

    with open('Final_Img/input_hex', 'w+') as f:
        np.savetxt(f,flat_hexfile,fmt="%x",delimiter='\n')
    '''

def image_sim():
    torch.backends.quantized.engine = 'qnnpack'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN_VSD(scale_factor=4).to(device)

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
            
            golden = np.zeros((25, 25, 16),dtype=int)



            for dim_x in range(25):
                for dim_y in range(25):
                    for dim_z in range(16):
                        offset_x = dim_z % 4
                        offset_y = (dim_z // 4) % 4
                        golden[dim_x][dim_y][dim_z] = preds[(dim_x * 4) + offset_x][(dim_y * 4) + offset_y]

            with open('Final_Img/Golden/golden{}'.format((5 * i) + j + 1),'w+') as f:
                np.savetxt(f,golden.flatten(),fmt="%x",delimiter='\n')

def image_post():

    resultPicture = pil_image.new('RGB', (500, 500), (0, 0, 0))

    for num in range(25):
        box = ((num // 5 % 5) * 100, (num % 5) * 100, (num // 5 % 5 + 1) * 100, (num % 5 + 1) * 100)
        image = pil_image.open(args_img).convert('RGB').resize((500,500)).crop(box)
        image_width = (image.width // args_scale) * args_scale
        image_height = (image.height // args_scale) * args_scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args_scale, hr.height //
                    args_scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args_scale, lr.height *
                            args_scale), resample=pil_image.BICUBIC)

        _, ycbcr = preprocess(bicubic, 'cpu')

        with open('Final_Img/OutputHex/tb{}/output'.format(num + 1), 'r') as f:
            temp = f.read().splitlines()

        hexfile = [int(i, 16) for i in temp]
        hexfile = np.array(hexfile, dtype='f').reshape(25, 25, 16)
        hexfile_ = np.zeros((100,100))

        for dim_x in range(25):
            for dim_y in range(25):
                for dim_z in range(16):
                    #offset_x = dim_z % 4
                    #offset_y = (dim_z // 4) % 4
                    offset_x = (dim_z // 4) % 4
                    offset_y = dim_z % 4
                    hexfile_[(dim_x * 4) + offset_x][(dim_y * 4) + offset_y] = hexfile[dim_x][dim_y][dim_z]

        output = np.array([hexfile_, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)

        resultPicture.paste(output, (box[0], box[1]))
        resultPicture.save("hardware.bmp")

    resultPicture.show()


if __name__ == "__main__":
    image_post()