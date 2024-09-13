import os

import cv2
import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

det_model_path = './model/buffalo_s/det_500m.onnx'
rec_model_path = './model/buffalo_s/w600k_mbf.onnx'


if (os.path.isfile(det_model_path)) and (os.path.isfile(rec_model_path)):
    det_model = model_zoo.get_model(det_model_path)
    rec_model = model_zoo.get_model(rec_model_path)
else:
    print('Pretrained model is not avaliable')
    quit()
print("hello")

det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)


