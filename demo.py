import cv2
import numpy as np
import time

print('OpenCV version: ', cv2.__version__)

def opencv_resize(imgfile):
    image = cv2.imread(imgfile)
    (h, w, _) = image.shape

    time1 = time.perf_counter()
    img_area_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_AREA)
    time2 = time.perf_counter()
    cv2.imwrite('./image/out_area_x4.bmp', img_area_x4)
    print('Resize INTER_AREA dur = %.4f ms' % ((time2-time1)*1000))

    time3 = time.perf_counter()
    img_cubic_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    time4 = time.perf_counter()
    cv2.imwrite('./image/out_cubic_x4.bmp', img_cubic_x4)
    print('Resize INTER_CUBIC dur = %.4f ms' % ((time4-time3)*1000))

    time5 = time.perf_counter()
    img_lanczos_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
    time6 = time.perf_counter()
    cv2.imwrite('./image/out_lanczos_x4.bmp', img_lanczos_x4)
    print('Resize INTER_LANCZOS4 dur = %.4f ms' % ((time6-time5)*1000))

def super_res_image(imgfile):
    image = cv2.imread(imgfile)
    (h, w, _) = image.shape
    for m in sr_models:
        model_path = "models/" + m
        m1, m2 =m.split('.pb')[0].split('_')
        model_name, model_scale = m1.lower(), int(m2[1:])
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel(model_name, model_scale)
        print('Start %s ...' % model_name)
        start = time.perf_counter()
        out_img = sr.upsample(image)
        dur = time.perf_counter() - start
        outfile = './image/out-%s_x%d.bmp' % (model_name, model_scale)
        cv2.imwrite(outfile, out_img)
        print('%dx upscaling with model = %s, output image = %s, dur = %.4f ms' % (model_scale, m, outfile, dur*1000))

def super_res_video(videofile, modelfile, frame_num=10):
    cap = cv2.VideoCapture(videofile)
    if not cap.isOpened():
        print("ERROR: cannot open vidoe file")
        return
    model_path = "models/" + modelfile
    m1, m2 =modelfile.split('.pb')[0].split('_')
    model_name, model_scale = m1.lower(), int(m2[1:])
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, model_scale)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret == False or frame_count > frame_num-1:
            break
        out_frame = sr.upsample(frame)
        cv2.imwrite('./out/%sx%d_%03d.bmp' % (model_name, model_scale, frame_count), out_frame)
        print('frame# %03d done' % frame_count)
        frame_count += 1
    cap.release()

sr_models = ['EDSR_x4.pb', 'ESPCN_x4.pb', 'FSRCNN_x4.pb', 'LapSRN_x4.pb']

imgfile = "image/test_input.png"
opencv_resize(imgfile)
super_res_image(imgfile)

videofile = 'video/test_input.mp4'
#super_res_video(videofile, 'LapSRN_x4.pb')


print('done')
