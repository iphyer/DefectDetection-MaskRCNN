# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.bounding_box import BoxList

class DefectDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, root, ann_file, transforms=None):
        super(DefectDetectionDataset, self).__init__(root, ann_file)
        self.transforms = transforms
       
    def __getitem__(self, idx):
        img, anno = super(DefectDetectionDataset, self).__getitem__(idx)
 
        
        # ---------------------
        image_info = anno[idx]['regions'] 
        masks = []
        classes = []
        boxes = []
       
        for item in len(image_info):
            boxes.append(item["shape_attributes"]['boundingbox'])
            all_points_x = item["shape_attributes"]['all_points_x']
            all_points_y = item["shape_attributes"]['all_points_y']
            for j in range(len(all_points_x)):
                mask.append(all_points_x[j])
                mask.append(all_points_y[j])
            masks.append(mask)
            classes.append(int(item["shape_attributes"]['name']))
            
        # -------------
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

       
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
