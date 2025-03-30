import argparse
import logging
import os

import cv2
import faiss
import insightface
import numpy as np
import pandas as pd
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

from constants import IMG_WIDTH

logging.basicConfig(level=logging.INFO, filename="./log/info.log",filemode="a", format="%(asctime)s %(levelname)s %(message)s")

ref_embeddings, ref_names = [], []

parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename", type=str,
                    help="CSV file of reference data", required= True)

parser.add_argument("-pd","--photodir", type=str,
                    help="photo directory", required= True)
args = parser.parse_args()

identities = [d for d in os.listdir(args.photodir) if os.path.isdir(os.path.join(args.photodir, d))]
name_list = []
img_path = []

# Get the identities 

for id in identities:
    path = os.path.join(args.photodir,id)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for f in files:
        file_path = os.path.join(path,f)
        name_list.append(id)
        img_path.append(file_path)


df = pd.DataFrame({'Name': name_list, 'file_path': img_path})
df.to_csv('output.csv', index=False)







det_model_path = './model/buffalo_s/det_500m.onnx'
rec_model_path = './model/buffalo_s/w600k_mbf.onnx'

det_model = model_zoo.get_model(det_model_path)
rec_model = model_zoo.get_model(rec_model_path)

det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)

logging.info(f"name_list: {len(name_list)}")
logging.info(f"img_list: {len(img_path)}")


if len(name_list) != len(img_path):
    logging.error('Length of name_list and img_list do not match')
    raise ValueError('The lenght of name_list and img_list do not match')

else:

    for name, img_file in zip(name_list, img_path):

        if os.path.isfile(str(img_file)):
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
            logging.error(f'File not found for {name}')
    logging.info(f'Length of ref_names: {len(name_list)}')
    logging.info(f'Length of embeddings: {len(ref_embeddings)}')
    if len(name_list) != len(ref_embeddings):
        logging.warning('Length of embedding and names do not match!!!')
    else:
        ref_embeddings = np.asarray(ref_embeddings)
        index =faiss.IndexFlatL2(ref_embeddings.shape[1])
        index.add(ref_embeddings)
        faiss.write_index(index,'./data/faiss_index.bin')
        logging.info(f'Quantity of image embeddings create and stored into vector store: {index.ntotal}')


        