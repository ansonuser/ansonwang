---
author: "Anson Wang"
date: "2025-03-19"
description: "Introduction of yolo family"
title: "Yolo Family (1)"
tags: [
    "Yolo",
    "Machine Learning",
    "Computer Vision"
]
math: true
---

## Dataset

COCO (Common Objects in Context, 80 object categories) is a large-scale dataset designed for object detection, segmentation, and image captioning and widely used for training and evaluating in computer vision. 

## COCO AP Metric:

- Precision (P): Measures how many predicted bounding boxes are correct.
- Recall (R): Measures how many actual objects were detected.
- Precision-Recall Curve: Plots precision against recall for different confidence thresholds.
- AP Calculation: The area under the Precision-Recall (PR) curve is computed as the AP score.

### Common Notations:
- AP (IoU=0.50:0.95) → Mean Average Precision (mAP)across IoUs (strictest).
- AP50 → AP at IoU = 0.50 (relaxed, detects if - objects are roughly correct).
- AP75 → AP at IoU = 0.75 (stricter, requires higher overlap).
- APsmall, APmedium, APlarge → AP scores for objects of different sizes.


## YOLOv1

Joseph Redmon et al. was published in CVPR 2016, as opposed to two-steps model like Fast R-CNN. 
Pre-trained the first conv layers on the ImageNet dataset. Finetuned last four layers + two fully connected layers on PASCAL VOC for object detection.

The image is divided into an S×S grid, each grid cell is responsible for predicting objects whose centers fall within that cell, B bounding boxes per grid cell for multiple objects in thee same region. Targets are assigned to grid cell by IoU.

{{< figure src="/ansonwang/images/yolov1_structure.png" alt="YOLO Arch" caption="Fig1. YOLOv1 Architecture. (src: Juan R. Terven, et al, 2024)" >}}

**Loss Function**

{{< figure src="/ansonwang/images/yolov1_loss_func.png">}}

Include localization error if it is predicted as an object (Confidence $$=P_{obj} \times IoU$$).

**Prediction**
- Bounding box coordinates: (x,y,w,h).
- Confidence score: Probability of an object’s presence combined with bounding-box accuracy (IoU).
- Class probabilities: Conditional probability of 
each class if an object exists in that cell.

=> output dimension: $$ S \times S \times(B \times 5 + C)$$, here S=7, B=2, C=20

Apply non-maximum suppression (NMS) to remove bad detections.

**Limitations**
- Struggles detecting small or overlapping objects(limited to B=2 boxes per cell).
- Poor handling of objects with unusual aspect ratios.(w, h are hard to learn without normalization)


## YOLOv2

{{< figure src="/ansonwang/images/yolov2_arch.png" alt="YOLOv2 Arch" caption="Fig2. YOLOv2 Architecture. (src: Chien-Yao Wang, et al, 2024)" >}}

**Improvements**:
- Fully convolutional layers(use channel as prediction) replace fully connected layers. 
- Anchor boxes (use a set of prior boxes)
- Direct Location prediction: Predict offsets (relative location to "grid cell"), see Fig2.
- Passthrough Layer: Converts 26×26×512 feature map into 13×13×2048 by moving spatial data into channel dimension.
- Multi-scale Training: Augmented with size.
- Backbone: Darknet-19 (Cutting layer 24->19, 3x3 conv in first layer, Systematic 1x1, BN, leakyrelu)




{{< figure src="/ansonwang/images/bounding_box_v2.png" alt="Bounding box v2" caption="Fig3. YOLOv2 Bounding Box. (src: Joseph Redmon et al., YOLOv3)" >}}




## YOLOv3

{{< figure src="/ansonwang/images/yolov3_arch.png" alt="YOLOv3 arch" caption="Fig4. YOLOv3-SPP architecture. It's a variant of original version (src: Juan R. Terven, et al, 2024)" >}}

- Objectness Prediction: Predict objectness directly (One object, one anchor (highest IoU))
- Remove softmax for classification. (Mult-labels but independent logistic is more powerful in pratice.)
- Scale up to Darknet53 (Replace max pooling with stride-2 convs, residual block)
- Spatial pyramid pooling (SPP): See Fig4, apply three different size(5, 9, 13) of maxpooling with padding to keep size unchanged for the channel-wise concatenation. 
- Multi-scale Predictions (FPN): See Fig4
    - SPP output goes through 5*CBLs hold spatial dimension still and recur channel size: 
    (1x1x512->3x3x1024->1x1x512->3x3x1024->1x1x512)
    - Upsample and infuse with 26x26x512 (medium object)
    - Upsample and infues with 52x52x256
    (small object)
- Bounding box priors with k-means


## References

[1] Chien-Yao Wang, et al, YOLOv1 to YOLOv10: The fastest and most accurate real-time object detection systems, 2024

[2] Juan R. Terven, et al, A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS, 2024

[3] Joseph Redmon, et al, You Only Look Once: Unified, Real-Time Object Detection, CVPR, 2015

[4] Joseph Redmon, et al, YOLO9000: Better, Faster, Stronger, CVPR, 2016

[5] Joseph Redmon, et al, YOLOv3: An Incremental Improvement, CVPR, 2018