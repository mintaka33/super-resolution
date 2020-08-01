import cv2
import numpy as np

print('OpenCV version: ', cv2.__version__)

image = cv2.imread("test3.png")
(h, w, _) = image.shape

def compare_resize(image, w, h):
    img_area_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_AREA)
    cv2.imwrite('out_area_x4.bmp', img_area_x4)
    img_cubic_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('out_cubic_x4.bmp', img_cubic_x4)
    img_lanczos_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('out_lanczos_x4.bmp', img_lanczos_x4)

model_path = "models/EDSR_x4.pb"
model_name, model_scale = 'edsr', 4
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel(model_name, model_scale)

cap = cv2.VideoCapture('test2.mp4')
if not cap.isOpened():
    print("ERROR: cannot open vidoe file")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if ret == False:
        break;
    outimg = sr.upsample(frame)
    cv2.imwrite('./out/edsrx4_%03d.bmp' % frame_count, outimg)
    print('frame# %03d processed' % frame_count)
    frame_count += 1

print('done')
