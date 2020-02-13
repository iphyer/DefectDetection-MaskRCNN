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
    confusionMatrix = np.zeros((4,4))
    gt_numbers = [0,0,0,0]
    IoUThreshold = 0.3
    locationError = 0
    pred_numbers = np.zeros(4)
    F1_avg = np.zeros(4) # for computing average of F1
    prec_avg = np.zeros(4)
    recall_avg = np.zeros(4)
    # Loop all testing images

    for image_name in os.listdir(testPath):
        print(image_name)
        correctList = np.zeros(4)
        gt_numbers_singleImage = np.zeros(4)  # record the number of gt defects in each image
        pred_numbers_singleImage = np.zeros(4) # record the number of predicted defects in each image
        gt_mask_list = [[],[],[],[]]
        #print(img)
        #image_name = "grid1_roi2_500kx_0p5nm_haadf1_0039.jpg"
        image = cv2.imread("../datasets/val/" + image_name)
        #imshow(image)
        # prepare gt mask
        catIds = coco.getCatIds()
        imgIds = coco.getImgIds(catIds=catIds )
        labels = list()
        allgtBG = np.zeros((1024,1024))
        allpredBG = np.zeros((1024,1024))
        with open('../datasets/annotations/val.json') as json_data:
            annotation = json.loads(json_data.read())
            images = annotation['images']
            imgId = 0
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

        # get the mask for each class
        for i in range(len(anns)):
            gt_numbers_singleImage[labels[i] - 1] += 1 
            if labels[i] == 1:
                gt_mask_list[0].append(coco.annToMask(anns[i]))
                gt_numbers[0] += 1 
                
            if labels[i] == 2:
                gt_mask_list[1].append(coco.annToMask(anns[i]))
                gt_numbers[1] += 1 
            if labels[i] == 3:
                gt_mask_list[2].append(coco.annToMask(anns[i]))
                gt_numbers[2] += 1 
            if labels[i] == 4:
                gt_mask_list[3].append(coco.annToMask(anns[i]))
                gt_numbers[3] += 1 
        #plt.imshow(gt_allMask)

        # begin predication
        # compute predictions
        predictions = coco_demo.run_on_opencv_image(image)
        cv2.imwrite(image_name, predictions)
        #imshow(predictions)
        mask, labels = coco_demo.get_predicted_mask_labels(image)
        #print(mask[0])
        # TODO : new_labels is the pred_labels to avoid labels for gt
        new_labels = np.zeros(len(labels))
        for i in range(len(labels)):
            new_labels[i] = labels[i].item()
        #print(new_labels)
        #pred_numbers = np.zeros(4)
        for i in new_labels:
           # print(type(i))
            item = int(i)
            pred_numbers[item-1] += 1
            pred_numbers_singleImage[item - 1] += 1
        
        # generate predict mask
        for i in range(len(new_labels)):
            maxIoU = 0
            maxLabel = 0
            currentPredMask = mask[i][0]
            allpredBG = allpredBG + currentPredMask
            for j in range(len(gt_mask_list)):
                for gtMask in gt_mask_list[j]:
                    union = np.count_nonzero(gtMask + currentPredMask)
                    intersection = np.count_nonzero((gtMask + currentPredMask) == 2)
                    tmpIoU = 1.0 * intersection / union
                    if tmpIoU > maxIoU:
                        maxIoU = tmpIoU
                        maxlabel = j + 1
            # loop all gt masks
            
            # check if location error
            if maxIoU > IoUThreshold :
                #print(new_labels[i] -1)
                #print(maxlabel - 1)
                if new_labels[i] == maxlabel:
                    correctList[maxlabel - 1] += 1
                confusionMatrix[int(new_labels[i] -1) ][maxlabel - 1] += 1
            else:
                locationError += 1
        for j in range(len(gt_mask_list)):
            for gtMask in gt_mask_list[j]:
                allgtBG = allgtBG + gtMask
        addAllBG = allgtBG + allpredBG
        BGIntersection = 1024*1024 - np.count_nonzero(addAllBG)
        UnionHelperMat = np.zeros((1024,1024))
        UnionHelperMat[np.where(allgtBG == 0)] = 1
        UnionHelperMat[np.where(allpredBG == 0)] = 1
        print("background intersection number:", BGIntersection)
        print("background union number:", np.count_nonzero(UnionHelperMat))
        print("gt non background:", np.count_nonzero(allgtBG > 0))
        print("predict non  background:", np.count_nonzero(allpredBG > 0))         
        print("=========================" + str(image_name) + "=========================")
        precision_list = np.zeros(4)
        recall_list = np.zeros(4)
        F1_list = np.zeros(4)
        for label in range(4):
            precision_list[label] = correctList[label] / pred_numbers_singleImage[label]
            recall_list[label] = correctList[label] / gt_numbers_singleImage[label]
            F1_list[label] = 2*(precision_list[label] * recall_list[label])/(recall_list[label]+precision_list[label])
            
            F1_avg[label] += F1_list[label]
            recall_avg[label] += recall_list[label]
            prec_avg[label] += precision_list[label]
        print("precision:", precision_list)
        print("recall", recall_list)
        print("f1",F1_list)
        
        
    
    print("location error rate is:", locationError)
    print(confusionMatrix)
    print("gt numbers", gt_numbers)    
    print("prediction numbers",pred_numbers)
    total_recall = []
    total_prec = []
    total_F1 = []
    for i in range(4):
        total_recall.append(confusionMatrix[i][i] / gt_numbers[i])
        total_prec.append(confusionMatrix[i][i] / pred_numbers[i])
        total_F1.append(2*total_recall[i]*total_prec[i] / (total_recall[i]+total_prec[i]))
        recall_avg[i] = recall_avg[i]/len(os.listdir(testPath))
        prec_avg[i] = prec_avg[i]/len(os.listdir(testPath))
        F1_avg[i] = F1_avg[i] / len(os.listdir(testPath))
    print(total_recall)
    print(total_prec)
    print(total_F1)
    
    
    print(recall_avg)
    print(prec_avg)
    print(F1_avg)
