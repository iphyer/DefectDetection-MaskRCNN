import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#import requests
from io import BytesIO
from PIL import Image
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
from coco import COCO
import os
import cv2
import json
def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

if __name__ == "__main__":    
    # this makes our figures bigger
    pylab.rcParams['figure.figsize'] = 20, 12
    config_file = "../configs/predict.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.85,
    )
    testPath = "../datasets/val"
    coco=COCO("../datasets/annotations/val.json")

    summaryIoU = [[],[],[],[],[]]
    summaryF1 = [[],[],[],[],[]]
    summaryP = [[],[],[],[],[]]
    summaryR = [[],[],[],[],[]]


    # Loop all testing images
    for image_name in os.listdir(testPath):
        print(image_name)
        #print(img)
        #image_name = "grid1_roi2_500kx_0p5nm_haadf1_0039.jpg"
        image = cv2.imread(testPath +"/" +image_name)
        #imshow(image)
        # prepare gt mask
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
        print(labels)
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        #plt.imshow(image)
        #coco.showAnns(anns)
        gt_allMask = np.zeros(coco.annToMask(anns[0]).shape)
        gt_type1Mask = np.zeros(coco.annToMask(anns[0]).shape)
        gt_type2Mask = np.zeros(coco.annToMask(anns[0]).shape)
        gt_type3Mask = np.zeros(coco.annToMask(anns[0]).shape)
        gt_type4Mask = np.zeros(coco.annToMask(anns[0]).shape)
        # get the mask for each class
        for i in range(len(anns)):
            if labels[i] == 1:
                gt_type1Mask += coco.annToMask(anns[i])
                gt_type1Mask[gt_type1Mask >= 2] = 1
            if labels[i] == 2:
                gt_type2Mask += coco.annToMask(anns[i])
                gt_type2Mask[gt_type2Mask >= 2] = 1
            if labels[i] == 3:
                gt_type3Mask += coco.annToMask(anns[i])
                gt_type3Mask[gt_type3Mask >= 2] = 1
            if labels[i] == 4:
                gt_type4Mask += coco.annToMask(anns[i])
                gt_type4Mask[gt_type4Mask >= 2] = 1
            gt_allMask += coco.annToMask(anns[i])
        #plt.imshow(gt_allMask)
        # begin predication
        # compute predictions
        predictions = coco_demo.run_on_opencv_image(image)
        cv2.imwrite(image_name, predictions)
        mask, labels = coco_demo.get_predicted_mask_labels(image)
        #print(mask[0])
        # TODO : new_labels is the pred_labels to avoid labels for gt
        new_labels = np.zeros(len(labels))
        for i in range(len(labels)):
            new_labels[i] = labels[i].item()
        #print(new_labels)
        pred_mask = np.zeros((1024,1024,4))
        # generate predict mask
        for i in range(len(new_labels)):
            if new_labels[i] == 1:
                pred_mask[:,:,0] += mask[i][0]
            if new_labels[i] == 2:
                pred_mask[:,:,1] += mask[i][0]
            if new_labels[i] == 3:
                pred_mask[:,:,2] += mask[i][0]
            if new_labels[i] == 4:
                pred_mask[:,:,3] += mask[i][0]
        #plt.imshow(pred_mask[:,:,0])
        type1_pred = [];
        type2_pred = [];
        type3_pred = [];
        type4_pred = [];
        allTypes_pred = [];
        class_ids = [1,2,3,4]
        mask1 = np.zeros((1024,1024)) # for this image
        mask2 = np.zeros((1024,1024))
        mask3 = np.zeros((1024,1024))
        mask4 = np.zeros((1024,1024))
        allmask = np.zeros((1024,1024))
        mask = pred_mask
        #print(class_ids)
        for j in range(len(class_ids)):
            this_channel = mask[:,:,j]
            class_id = class_ids[j]
        #     print(np.count_nonzero(this_channel))
            if class_id == 1:
                mask1 = mask1 + this_channel
                mask1[mask1 >= 2] = 1
            elif class_id == 2:
                mask2 = mask2 + this_channel
                mask2[mask2 >= 2] = 1
            elif class_id == 3:
                mask3 = mask3 + this_channel
                mask3[mask3 >= 2] = 1
            else:
                mask4 = mask4 + this_channel
                mask4[mask4 >= 2] = 1
            allmask = allmask + this_channel
            allmask[allmask >= 2] = 1

        #print(np.count_nonzero(mask1)/(1024*1024))        
        type1_pred.append(mask1)
        type2_pred.append(mask2)
        type3_pred.append(mask3)
        type4_pred.append(mask4)
        allTypes_pred.append(allmask)
        #print(len(type1_pred))
        type1_union = []
        type2_union = []
        type3_union = []
        type4_union = []
        alltypes_union = []
        type1_intersection = []
        type2_intersection = []
        type3_intersection = []
        type4_intersection = []
        alltypes_intersection = []
        type1_iou = []
        type2_iou = []
        type3_iou = []
        type4_iou = []
        alltypes_iou = []
        type1_precision = []
        type2_precision = []
        type3_precision = []
        type4_precision = []
        alltypes_precision = []
        type1_recall = []
        type2_recall = []
        type3_recall = []
        type4_recall = []
        alltypes_recall = []

        type1_F1 = []
        type2_F1 = []
        type3_F1 = []
        type4_F1 = []
        alltypes_F1 = []
        i = 0
        # for i in range(): 
        gt_mask1 = gt_type1Mask  #the first type gt mask of this ith image
        gt_mask2 = gt_type2Mask 
        gt_mask3 = gt_type3Mask 
        gt_mask4 = gt_type4Mask
        gt_allmask = gt_allMask 

        pred_mask1 = type1_pred[i]
        pred_mask2 = type2_pred[i]
        pred_mask3 = type3_pred[i]
        pred_mask4 = type4_pred[i]
        pred_allmask = allTypes_pred[i]

        type1_union.append(np.count_nonzero(gt_mask1+pred_mask1))
        type2_union.append(np.count_nonzero(gt_mask2+pred_mask2))
        type3_union.append(np.count_nonzero(gt_mask3+pred_mask3))
        type4_union.append(np.count_nonzero(gt_mask4+pred_mask4))
        alltypes_union.append(np.count_nonzero(gt_allmask+pred_allmask))

        type1_intersection.append(np.count_nonzero((gt_mask1+pred_mask1) == 2))
        type2_intersection.append(np.count_nonzero((gt_mask2+pred_mask2) == 2))
        type3_intersection.append(np.count_nonzero((gt_mask3+pred_mask3) == 2))
        type4_intersection.append(np.count_nonzero((gt_mask4+pred_mask4) == 2))
        alltypes_intersection.append(np.count_nonzero((gt_allmask+pred_allmask) == 2)) 

        type1_iou.append(type1_intersection[i]/type1_union[i])
        type2_iou.append(type2_intersection[i]/type2_union[i])
        type3_iou.append(type3_intersection[i]/type3_union[i])
        type4_iou.append(type4_intersection[i]/type4_union[i])
        alltypes_iou.append(alltypes_intersection[i]/alltypes_union[i])
        
        type1_precision.append(type1_intersection[i]/np.count_nonzero((pred_mask1)))


        type2_precision.append(type2_intersection[i]/np.count_nonzero((pred_mask2)))
        if np.count_nonzero((pred_mask3)) > 0:
            type3_precision.append(type3_intersection[i]/np.count_nonzero((pred_mask3)))
        else:
            type3_precision.append(0)
        if np.count_nonzero((pred_mask4)) > 0:
            type4_precision.append(type4_intersection[i]/np.count_nonzero((pred_mask4)))
        else:
            type4_precision.append(0)
            
        if np.count_nonzero(pred_allmask)> 0:
            alltypes_precision.append(alltypes_intersection[i] / np.count_nonzero(pred_allmask))
        else:
            alltypes_precision.append(0)
        
        type1_recall.append(type1_intersection[i]/np.count_nonzero((gt_mask1)))

        type2_recall.append(type2_intersection[i]/np.count_nonzero((gt_mask2)))
        type3_recall.append(type3_intersection[i]/np.count_nonzero((gt_mask3)))
        if np.count_nonzero((gt_mask4)) > 0:
            type4_recall.append(type4_intersection[i]/np.count_nonzero((gt_mask4)))
        else:
            type4_recall.append(0)
        alltypes_recall.append(alltypes_intersection[i] / np.count_nonzero(gt_allmask))
        if (type1_recall[i] + type1_precision[i]) > 0:
            type1_F1.append(2*(type1_recall[i]* type1_precision[i]) / (type1_recall[i] + type1_precision[i]))
        else:
            type1_F1.append(0)
        if (type1_recall[i] + type1_precision[i]) > 0:
            type2_F1.append(2*(type2_recall[i] * type2_precision[i]) / (type2_recall[i] + type2_precision[i]))
        else:
            type2_F1.append(0)
        if (type3_recall[i] + type3_precision[i]) > 0:
            type3_F1.append(2*(type3_recall[i] * type3_precision[i]) / (type3_recall[i] + type3_precision[i]))
        else:
            type3_F1.append(0)
        if (type4_recall[i] + type4_precision[i]) > 0:
            type4_F1.append(2*(type4_recall[i] * type4_precision[i]) / (type4_recall[i] + type4_precision[i]))
        else:
            type4_F1.append(0)
        alltypes_F1.append(2*(alltypes_recall[i] * alltypes_precision[i]) / (alltypes_recall[i] + alltypes_precision[i]))
        # Output current results
        print("The performance of image %s"%image_name)
        print("======================= IoU ================================")
        print('111 loop iou')
        print(type1_iou)
        print('blackdot iou')
        print(type2_iou)
        print('100 loop iou')
        print(type3_iou)
        print('100 type 2 iou')
        print(type4_iou)
        print('all types iou')
        print(alltypes_iou)
        # append results
        summaryIoU[0].append(type1_iou[0])
        summaryIoU[1].append(type2_iou[0])
        summaryIoU[2].append(type3_iou[0])
        summaryIoU[3].append(type4_iou[0])
        summaryIoU[4].append(alltypes_iou[0])
        print("======================= F1 ================================")
        print('111 loop F1')
        print(type1_F1)
        print('blackdot F1')
        print(type2_F1)
        print('100 loop F1')
        print(type3_F1)
        print('100 type 2 F1')
        print(type4_F1)
        print('all types F1')
        print(alltypes_F1)
        # append results
        summaryF1[0].append(type1_F1[0])
        summaryF1[1].append(type2_F1[0])
        summaryF1[2].append(type3_F1[0])
        summaryF1[3].append(type4_F1[0])
        summaryF1[4].append(alltypes_F1[0])
        print("======================= P ================================")
        print('111 loop precision')
        print(type1_precision)
        print('blackdot precision')
        print(type2_precision)
        print('100 loop precision')
        print(type3_precision)
        print('100 type 2 precision')
        print(type4_precision)
        print('all types precision')
        print(alltypes_precision)
        summaryP[0].append(type1_precision[0])
        summaryP[1].append(type2_precision[0])
        summaryP[2].append(type3_precision[0])
        summaryP[3].append(type4_precision[0])

        summaryP[4].append(alltypes_precision[0])
        print("======================= R ================================")
        print('111 loop recall')
        print(type1_recall)
        print('blackdot recall')
        print(type2_recall)
        print('100 loop recall')
        print(type3_recall)
        print('100 loop recall')
        print(type4_recall)
        print('all types recall')
        print(alltypes_recall)
        summaryR[0].append(type1_recall[0])
        summaryR[1].append(type2_recall[0])
        summaryR[2].append(type3_recall[0])
        summaryR[3].append(type4_recall[0])
        summaryR[4].append(alltypes_recall[0])
