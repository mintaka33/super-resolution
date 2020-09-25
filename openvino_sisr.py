import sys
import os
import cv2
import numpy as np
from openvino.inference_engine import IECore

def dump2file(a, i=0):
    with open('out_%02d.txt'%i, 'wt') as f:
        h, w = a.shape
        for y in range(h):
            line = []
            for x in range(w):
                line.append('%8.4f'%a[y][x])
            line.append('\n')
            f.write(', '.join(line))

device = 'CPU'
video_file = 'test.265'
model_xml = 'model/single-image-super-resolution-1032.xml'
model_bin = 'model/single-image-super-resolution-1032.bin'

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name=device)
supported_layers = ie.query_network(net, device)

not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
if len(not_supported_layers) != 0:
    print('Unsupported layers:', not_supported_layers)
    exit()

input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
net.batch_size = 1

n, c, h, w = net.input_info[input_blob].input_data.shape
images = np.ndarray(shape=(n, c, h, w))

img = cv2.imread('test.png')
img2 = img.astype(np.float32)/255
images[0] = img2.transpose((2, 0, 1))

res = exec_net.infer(inputs={input_blob: images})
data = res[out_blob]

#dump2file(data[0][1], 0)

d = [ np.clip(c*255, 0, 255) for c in data[0]]

dump2file(d[0], 1)

R = d[0].astype(np.int8)
G = d[1].astype(np.int8)
B = d[2].astype(np.int8)

cv2.imwrite('out_r.bmp', R)
cv2.imwrite('out_g.bmp', G)
cv2.imwrite('out_b.bmp', B)

outimg = cv2.merge((R, G, B))
cv2.imwrite('out.bmp', outimg)

print('done')