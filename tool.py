import cv2
import numpy as np

def gen_rgpb(imgfile):
    img = cv2.imread(imgfile)
    b, g, r = cv2.split(img)
    with open('tmp.rgbp', 'wb') as f:
        for c in [r, g, b]:
            c.tofile(f)

#gen_rgpb('tmp.bmp')

with open("tmp.rgbp", "rb") as f:
    data = np.fromfile(f, "uint8")

data = data.reshape((3, 270, 480))
_, h, w = data.shape

R = data[0,:,:]
G = data[1,:,:]
B = data[2,:,:]

Y =  0.257 * R + 0.504 * G + 0.098 * B +  16
U = -0.148 * R - 0.291 * G + 0.439 * B + 128
V =  0.439 * R - 0.368 * G - 0.071 * B + 128

Y = Y.astype(np.uint8)
U = U.astype(np.uint8)
V = V.astype(np.uint8)

with open('tmp.yuv', 'wb') as f:
    for p in [Y, U, V]:
        p.tofile(f)

UV = np.zeros((int(h/2), w)).astype(np.uint8)
U2 = U[::2, 0::2]
V2 = V[::2, 0::2]
UV[:, 0::2] = U2
UV[:, 1::2] = V2

with open('tmp.nv12', 'wb') as f:
    Y.tofile(f)
    UV.tofile(f)

print('done')