import logging
import os
import time

import cv2
import faiss
import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

logging.basicConfig(level=logging.INFO, filename="./log/info.log",filemode="a", format="%(asctime)s %(levelname)s %(message)s")

# Open the default camera
cam = cv2.VideoCapture(0)

# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
logging.info(f"Frame width: {frame_width}")
logging.info(f"Frame height: {frame_height}")

det_model_path = './model/buffalo_s/det_500m.onnx'
rec_model_path = './model/buffalo_s/w600k_mbf.onnx'


if (os.path.isfile(det_model_path)) and (os.path.isfile(rec_model_path)):
    det_model = model_zoo.get_model(det_model_path)
    rec_model = model_zoo.get_model(rec_model_path)
else:
    logging.info('Pretrained model is not avaliable')
    quit()


det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)

def draw_faces(image,
               faces,
               draw_landmarks=False,
               show_confidence=False,
               names=[],
               show_names=False):
    
    for i, face in enumerate(faces):
        color = (0, 255, 255)
        thickness = 2
        cv2.rectangle(image, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), color, thickness, cv2.LINE_AA)

        if draw_landmarks:
            landmarks = face['landmarks']
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)

        if show_confidence:
            confidence = face[4]
            confidence = "{:.2f}".format(confidence)
            position = (int(face[0]), int(face[1] - 10))
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)

        if show_names and len(faces)==len(names):

            position = (int(face[0]), int(face[1] - 25))
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            cv2.putText(image, names[i], position, font, scale, color, thickness, cv2.LINE_AA)            

    return image



cur_frametime, prev_frametime = 0, 0

while True:


    ret, frame = cam.read()

    # Write the frame to the output file
    # out.write(frame)

    bboxes, kpss = det_model.detect(frame, max_num=0, metric='default')
    frame = draw_faces(frame, bboxes, show_confidence=True)


    # Display the captured frame
    cv2.imshow('Camera', frame)



    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    cur_frametime = time.time()
    interval = cur_frametime - prev_frametime
    framerate = int(1/ interval)
    prev_frametime = cur_frametime
    print(framerate)


# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()