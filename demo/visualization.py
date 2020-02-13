# from http://cocodataset.org/#explore?id=345434
import cv2
import json
from predictor import COCODemo
import numpy as np
from coco import COCO

image_name = "0501_300kx_1nm_clhaadf3_0006.jpg"
image = cv2.imread("../datasets/val/" + image_name)

coco=COCO("../datasets/annotations/val.json")
catIds = coco.getCatIds()
imgIds = coco.getImgIds(catIds=catIds );
labels = list()
with open('../datasets/annotations/val.json') as json_data:
    annotation = json.loads(json_data.read())
    images = annotation['images']
    for i in range(len(images)):
        if(images[i]["file_name"] == image_name):
            imgId = images[i]["id"]
    
    seg = annotation['annotations']
    for i in range(len(seg)):
        if seg[i]['image_id'] == imgId:
            labels.append(seg[i]['category_id'])
    
img = coco.loadImgs(imgId)[0]
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

