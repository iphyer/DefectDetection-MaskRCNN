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
    iou_threshold = 0.1
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
    allBG_F1 = [[],[],[],[],[]]

    pixel_acc_list = [[],[],[],[],[]]
    pixel_recall_list = [[],[],[],[],[]]
    pixel_precision_list = [[],[],[],[],[]]
    pixel_IOU_list = [[],[],[],[],[]]
    bbox_iou_list = []
    bbox_threshold = 0.5
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
        gt_labels = list()
        bboxes = []
        with open('../datasets/annotations/val.json') as json_data:
            annotation = json.loads(json_data.read())
            images = annotation['images']
            for i in range(len(images)):
                if(images[i]["file_name"] == image_name):
                    imgId = images[i]["id"]

            seg = annotation['annotations']
            for i in range(len(seg)):
                if seg[i]['image_id'] == imgId:
                    gt_labels.append(seg[i]['category_id'])
                    bboxes.append(seg[i]['bbox'])    
        
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
            if gt_labels[i] == 1:
                gt_type1Mask += coco.annToMask(anns[i])
                gt_type1Mask[gt_type1Mask >= 2] = 1
            if gt_labels[i] == 2:
                gt_type2Mask += coco.annToMask(anns[i])
                gt_type2Mask[gt_type2Mask >= 2] = 1
            if gt_labels[i] == 3:
                gt_type3Mask += coco.annToMask(anns[i])
                gt_type3Mask[gt_type3Mask >= 2] = 1
            if gt_labels[i] == 4:
                gt_type4Mask += coco.annToMask(anns[i])
                gt_type4Mask[gt_type4Mask >= 2] = 1
            gt_allMask += coco.annToMask(anns[i])
        gt_mask_list = [[],[],[],[]]
        for i in range(len(anns)):
            if gt_labels[i] == 1:
                gt_mask_list[0].append(coco.annToMask(anns[i]))
 
            if gt_labels[i] == 2:
                gt_mask_list[1].append(coco.annToMask(anns[i]))
            if gt_labels[i] == 3:
                gt_mask_list[2].append(coco.annToMask(anns[i]))
            if gt_labels[i] == 4:
                gt_mask_list[3].append(coco.annToMask(anns[i]))

        #plt.imshow(gt_allMask)
        # begin predication
        # compute predictions
        predictions = coco_demo.run_on_opencv_image(image)
        cv2.imwrite(image_name, predictions)
        mask, labels = coco_demo.get_predicted_mask_labels(image) # mask is predicted mask
        #print(mask[0])
        print(len(labels))
        print(len(mask))
        print(len(anns))
        # TODO : new_labels is the pred_labels to avoid labels for gt
        new_labels = np.zeros(len(labels))
        for i in range(len(labels)):
            new_labels[i] = labels[i].item()
        #print(new_labels)
        pred_mask = np.zeros((1024,1024,4))
        

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
        
        mask1 = np.zeros((1024,1024)) # 111 prediction for this image
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
        
        
        # 111 loop
        TP = np.zeros(5)
        TN = np.zeros(5)
        FP = np.zeros(5)
        FN = np.zeros(5)
        
        masks_list = [mask1,mask2,mask3,mask4,allmask]
        gt_mask_list = [gt_type1Mask,gt_type2Mask,gt_type3Mask,gt_type4Mask,gt_allMask]
        for i in range(5):
            TP[i] = np.count_nonzero((masks_list[i] + gt_mask_list[i]) >= 2)
            TN[i] = 1024*1024 - np.count_nonzero(masks_list[i] + gt_mask_list[i])
            FP_helper = masks_list[i] * 2
            FP_sum = FP_helper + gt_mask_list[i]
            FP[i] = np.count_nonzero(FP_sum == 2)
            FN[i] = np.count_nonzero(FP_sum == 1)
            pixel_acc = (TP[i] + TN[i])/(1024*1024)#(TP[i] + TN[i])/(TP[i] + TN[i] + FN[i] + FP[i])
            print("all TP + TN + FN + FP", TP[i] + TN[i] + FN[i] + FP[i])
            if TP[i] == 0:
                pixel_recall = 0
                pixel_prec = 0
                pixel_IOU = 0
            else:

                pixel_recall = TP[i]/np.count_nonzero(gt_mask_list[i])
                pixel_prec = TP[i]/np.count_nonzero(masks_list[i])
                pixel_IOU = TP[i]/(1024*1024-TN[i])
            if pixel_recall == float('nan'):
                pixel_recall = 0
            if pixel_recall == 0 or pixel_recall == float('nan'):
                pixel_prec = 0
            pixel_acc_list[i].append(pixel_acc)
            pixel_recall_list[i].append(pixel_recall)
            pixel_precision_list[i].append(pixel_prec)
            pixel_IOU_list[i].append(pixel_IOU)
            print("Recall: Type " + str(i) + ":" + str(pixel_recall))
            print("Precision: Type " + str(i) + ":" + str(pixel_prec))
            print("IoU: Type " + str(i) + ":" + str(pixel_IOU))
            print("Type " + str(i) + ":" + str(pixel_acc))
            
        pred_bboxes, pred_labels = coco_demo.get_bbox(image)
#         print("length of predicted boxes:", pred_bboxes.shape)
#         print("length of gt boxes:", len(bboxes))
#         print("length of labels:", len(labels))
        
#         gt_bbox_map = np.zeros((1024,1024,len(bboxes))) # each channel is one gt defect, pixel level => IoU
#         pred_bbox_map = np.zeros((1024,1024,len(pred_bboxes)))
#         for i in range(len(bboxes)):
            
#             start_i = int(bboxes[i][0])
#             end_i = int(bboxes[i][2]+bboxes[i][0])
#             start_j = int(bboxes[i][1])
#             end_j = int(bboxes[i][1]+bboxes[i][3])
#             gt_bbox_map[start_i:end_i,start_j:end_j,i] = 1
        
#         for i in range(len(pred_bboxes)):
            
#             start_i = int(pred_bboxes[i][0])
#             end_i = int(pred_bboxes[i][1])#int(pred_bboxes[i][2]+pred_bboxes[i][0])
#             start_j = int(pred_bboxes[i][2])#int(pred_bboxes[i][1])
#             end_j = int(pred_bboxes[i][3])#int(pred_bboxes[i][1]+pred_bboxes[i][3])
#             pred_bbox_map[start_i:start_j,end_i:end_j,i] = 1
        
#         index_list = []  # store which pred correspond to which gt
#         iou_list = []
#         label_list = []
#         for i in range(len(pred_bboxes)):
#             max_threshold = 0
#             max_label = 0
#             max_index = -1
#             for j in range(len(bboxes)):
                
#                 bbox_interaction = np.count_nonzero((pred_bbox_map[:,:,i]+gt_bbox_map[:,:,j]) >= 2)
#                 bbox_union = np.count_nonzero(pred_bbox_map[:,:,i]+gt_bbox_map[:,:,j])
#                 bbox_iou = bbox_interaction/bbox_union
#                 if bbox_iou > max_threshold:
#                     max_threshold = bbox_iou
#                     max_index = j
#                     max_label = gt_labels[j]
           
#             index_list.append(max_index)
#             iou_list.append(max_threshold)
#             label_list.append(max_label)
#         for i in range(len(pred_bboxes)):
#             if index_list[i] == -1:
#                 continue
#             if iou_list[i] > iou_threshold:
#                 if label_list[i] == pred_labels[i]:
#                     bbox_iou_list.append(iou_list[i])
#         print(index_list)
    for i in range(5):    
        print(sum(pixel_acc_list[i]) / len(os.listdir(testPath))) 
        print(sum(pixel_recall_list[i]) / len(os.listdir(testPath))) 
        print(sum(pixel_precision_list[i]) / len(os.listdir(testPath))) 
        print(sum(pixel_IOU_list[i]) / len(os.listdir(testPath))) 
#         print(sum(bbox_iou_list)/len(bbox_iou_list))
        print("===============================================")
