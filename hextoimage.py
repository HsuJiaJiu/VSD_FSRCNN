import numpy as np
from numpy.core.arrayprint import printoptions
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
import PIL.Image as pil_image


def singleHex():

    with open('FeatureMap_hex/Layer8/conv_out', 'r') as f:
        #hexfile = np.array(f.read().splitlines(), dtype='f').reshape(100, 100)
        temp = f.read().splitlines()

    hexfile = [int(i, 16) for i in temp]
    hexfile = np.array(hexfile, dtype='f').reshape(100, 100)

    print(hexfile)

    ycbcr = np.load('FeatureMap_hex/ycbcr.npy')

    output = np.array([hexfile, ycbcr[..., 1], ycbcr[..., 2]]
                      ).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save('test1.bmp')


def spiltHex():
    '''
    with open('FeatureMap_hex/Layer8/conv_out_spilt', 'r') as f:
        temp = f.read().splitlines()
    '''
    with open('sram', 'r') as f:
        temp = f.read().splitlines()

    hexfile = [int(i, 16) for i in temp]
    hexfile = np.array(hexfile, dtype='f').reshape(16, 25, 25)

    hexfile_ = np.zeros((100,100))

    #Rearrange
    for i in range(16):
        init_u = (i % 4)
        init_v = ((i // 4) % 4)
        for j in range(25):
            for k in range(25):
                target_u = init_u + 4 * k
                target_v = init_v + 4 * j
                if (target_u < 100 and target_v < 100):
                    hexfile_[target_v][target_u] = hexfile[i][j][k]

    ycbcr = np.load('FeatureMap_hex/ycbcr.npy')

    output = np.array([hexfile_, ycbcr[..., 1], ycbcr[..., 2]]
                      ).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save('sram.bmp')


if __name__ == "__main__":
    spiltHex()
