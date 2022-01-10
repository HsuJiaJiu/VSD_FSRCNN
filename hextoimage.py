import numpy as np
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
import PIL.Image as pil_image


def singleHex():

    with open('FeatureMap/Layer8/conv_out', 'r') as f:
        hexfile = np.array(f.read().splitlines(), dtype='f').reshape(100, 100)
        print(hexfile.dtype)

    ycbcr = np.load('FeatureMap/ycbcr.npy')

    output = np.array([hexfile, ycbcr[..., 1], ycbcr[..., 2]]
                      ).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save('test.bmp')


def spiltHex():
    pass


if __name__ == "__main__":
    singleHex()
