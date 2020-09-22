import cv2
import numpy as np
import os

width, height = 480, 270

def gen_rgpb():
    os.system('ffmpeg -y -f lavfi -i testsrc2 -s %dx%d -vframes 1 tmp.bmp' % (width, height))
    img = cv2.imread('tmp.bmp')
    b, g, r = cv2.split(img)
    with open('tmp.rgbp', 'wb') as f:
        for c in [r, g, b]:
            c.tofile(f)

gen_rgpb()

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

with open('tmp.out.yuv', 'wb') as f:
    for p in [Y, U, V]:
        p.tofile(f)
os.system('ffmpeg -y -pix_fmt yuv444p -s %dx%d -i tmp.out.yuv tmp.out.yuv444.bmp' % (width, height))

UV = np.zeros((int(h/2), w)).astype(np.uint8)
U2 = U[::2, 0::2]
V2 = V[::2, 0::2]
UV[:, 0::2] = U2
UV[:, 1::2] = V2

with open('tmp.out.nv12', 'wb') as f:
    Y.tofile(f)
    UV.tofile(f)
os.system('ffmpeg -y -pix_fmt nv12 -s %dx%d -f rawvideo -i tmp.out.nv12 tmp.out.nv12.bmp' % (width, height))

print('done')