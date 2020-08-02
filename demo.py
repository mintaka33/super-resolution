import cv2
import numpy as np

print('OpenCV version: ', cv2.__version__)

def opencv_resize(imgfile):
    image = cv2.imread(imgfile)
    (h, w, _) = image.shape
    img_area_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./image/out_area_x4.bmp', img_area_x4)
    img_cubic_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('./image/out_cubic_x4.bmp', img_cubic_x4)
    img_lanczos_x4 = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('./image/out_lanczos_x4.bmp', img_lanczos_x4)

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
        out_img = sr.upsample(image)
        outfile = './image/out-%s_x%d.bmp' % (model_name, model_scale)
        cv2.imwrite(outfile, out_img)
        print('%dx upscaling with model = %s, output image = %s' % (model_scale, m, outfile))

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
super_res_video(videofile, 'LapSRN_x4.pb')


print('done')
