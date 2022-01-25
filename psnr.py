import PIL.Image as pil_image
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr

path = 'data/lenna.bmp'

if __name__ == '__main__':
    image = pil_image.open(path).resize((500,500)).convert('RGB')
    image_fsrcnn = pil_image.open(path.replace('.', '_spilt_fsrcnn_x4.')).resize((500,500)).convert('RGB')
    image_bicubic = pil_image.open(path.replace('.', '_bicubic_x4.')).resize((500,500)).convert('RGB')

    image, _ = preprocess(image, 'cpu')
    image_fsrcnn, _ = preprocess(image_fsrcnn, 'cpu')
    image_bicubic, _ = preprocess(image_bicubic, 'cpu')

    psnr = calc_psnr(image, image_bicubic)
    print(psnr)
    psnr = calc_psnr(image, image_fsrcnn)
    print(psnr)

