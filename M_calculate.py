import collections
from math import inf
from tempfile import tempdir
import torch
import numpy as np


def joke():
    with open('FeatureMap/Layer1/conv_in', 'r') as f:
        fin = torch.from_numpy(
            np.array(f.read().split(), dtype='f').reshape(1, 1, 25, 25))

    with open('FeatureMap/Layer1/conv_out', 'r') as f:
        fout = torch.from_numpy(
            np.array(f.read().split(), dtype='f').reshape(1, 56, 25, 25))

    Qbest = torch.load('Qbest.pth', map_location=lambda storage, loc: storage)

    temp_weight = collections.OrderedDict()
    temp_weight['weight'] = torch.zeros(56, 1, 5, 5)

    for i in range(56):
        for j in range(5):
            for k in range(5):
                temp_weight['weight'][i][0][j][k] = torch.int_repr(
                    Qbest['first_part.0.weight'][i][0][j][k]).item()

    conv = torch.nn.Conv2d(1, 56, kernel_size=5,
                           stride=1, padding=2, bias=False)
    conv.load_state_dict(temp_weight)
    relu = torch.nn.ReLU(inplace=False)

    with torch.no_grad():
        inference = conv(fin)
        inference = relu(inference)

    M_sum = 0
    count = 0
    for i in range(56):
        for j in range(25):
            for k in range(25):
                if(fout[0][i][j][k] != 0):
                    M_sum += (inference[0][i][j][k] / fout[0][i][j][k])
                    count += 1

    M = (M_sum) / count  # 理論M值
    M = 1 / M

    for i in range(20):
        print('M0 = ' + str(M * (2**i)) + '\tS = ' + str(i))


if __name__ == "__main__":
    joke()
