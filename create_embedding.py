import argparse
import logging
import os

import cv2
import insightface
import pandas as pd
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

from constants import IMG_WIDTH

logging.basicConfig(level=logging.INFO, filename="./log/info.log",filemode="a", format="%(asctime)s %(levelname)s %(message)s")

ref_embeddings, ref_names = [], []

parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename", type=str,
                    help="CSV file of reference data", required= True)

args = parser.parse_args()
# Load the CSV containing the names and their respective photo filename
df = pd.read_csv(args.filename, header=None)

# Get the names and images to be embedded into list
name_list = df[0].to_list()
img_list = df[1].to_list()


det_model_path = './model/buffalo_s/det_500m.onnx'
rec_model_path = './model/buffalo_s/w600k_mbf.onnx'

det_model = model_zoo.get_model(det_model_path)
rec_model = model_zoo.get_model(rec_model_path)

det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)

logging.info(f"name_list: {len(name_list)}")
logging.info(f"img_list: {len(img_list)}")


if len(name_list) != len(img_list):
    logging.error('Length of name_list and img_list do not match')
    raise ValueError('The lenght of name_list and img_list do not match')

else:

    for name, img_file in zip(name_list, img_list):

        if os.path.isfile(img_file):
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            (h, w) = img.shape[:2]
            aspect_ratio = w / h
            new_height = int(IMG_WIDTH /aspect_ratio)
            img = cv2.resize(img,(IMG_WIDTH, new_height))
            bboxes, kpss = det_model.detect(img, max_num=0, metric='default')
            # img = draw_faces(img,bboxes, show_confidence=True)

            # Check that the image only a single face is detected
            if bboxes.shape[0]==1:

                bbox = bboxes[0,:4]     # The bounding box coordinates
                confidence_score = bboxes[0,4]  # The confidence score
                kps = kpss[0]   

                face = Face(bbox=bbox, kps = kps, det_score =confidence_score)
                rec_model.get(img, face)

                ref_embeddings.append(face.normed_embedding)
                ref_names.append(name)
            
            else:
                logging.warning(f'More than one face detect in file:{img_file}')
        else:
            logging.error
    logging.info(f'Len of ref_names: {len(name_list)}')
    logging.info(f'len of embeddings: {len(ref_embeddings)}')
        