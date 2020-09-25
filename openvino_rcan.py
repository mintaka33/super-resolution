import sys
import os
import cv2
import numpy as np
from openvino.inference_engine import IECore

def dump2file(a, i=0, datatype='float'):
    filename = 'test2.dump.%02d.txt'%i
    with open(filename, 'wt') as f:
        h, w = a.shape
        for y in range(h):
            line = []
            for x in range(w):
                d = '%12.4f'%a[y][x] if datatype == 'float' else '%04d'%a[y][x]
                line.append(d)
            line.append('\n')
            f.write(', '.join(line))
    print('dump data in %s' % filename)

device = 'CPU'
input_file = 'test2.png'
model_xml = 'model/rcan_360x640_rgbp.xml'
model_bin = 'model/rcan_360x640_rgbp.bin'

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name=device)
supported_layers = ie.query_network(net, device)

not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
if len(not_supported_layers) != 0:
    print('Unsupported layers:', not_supported_layers)
    exit()

input_layer = next(iter(net.input_info))
out_layer = next(iter(net.outputs))

net.batch_size = 1

n, c, h, w = net.input_info[input_layer].input_data.shape
print('INFO: [NCHW] = ', n, c, h, w)
images = np.ndarray(shape=(n, c, h, w))

img = cv2.imread(input_file)
img2 = img.astype(np.float32)
images[0] = img2.transpose((2, 0, 1))

# do inference
res = exec_net.infer(inputs={input_layer: images})
data = res[out_layer]

dump2file(data[0][1], 0)

d = [ np.clip(c, 0, 255) for c in data[0]]

dump2file(d[0], 1)

R = d[0].astype(np.uint8)
G = d[1].astype(np.uint8)
B = d[2].astype(np.uint8)

dump2file(R, 2, 'int')
cv2.imwrite('test2.out.R.bmp', R)

outimg = cv2.merge((R, G, B))
cv2.imwrite('test2.out.png', outimg)

print('done')