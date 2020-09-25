import sys
import os
import cv2
import numpy as np
from openvino.inference_engine import IECore

def dump2file(a, i=0, datatype='float'):
    filename = 'test3.dump.%02d.txt'%i
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
input_file = 'test3.png'
model_xml = 'model/rcan_enc_fp32.xml'
model_bin = 'model/rcan_enc_fp32.bin'

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

img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
images[0] = img.astype(np.float32).reshape((1, h, w))

# do inference
res = exec_net.infer(inputs={input_layer: images})
data = res[out_layer]

dump2file(data[0][0], 0)

d = np.clip(data[0][0], 0, 255)
Y = d.astype(np.uint8)
dump2file(Y, 1, 'int')

cv2.imwrite('test3.out.Y.bmp', Y)

print('done')