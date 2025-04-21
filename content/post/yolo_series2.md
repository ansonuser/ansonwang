---
author: "Anson Wang"
date: "2025-03-23"
description: "Introduction of yolo family"
title: "Yolo Family (2)"
tags: [
    "Yolo",
    "Machine Learning",
    "Computer Vision"
]
math: true
---


## YOLOv4

Joseph Redmon stopped researching on computer vision. Alexey took over and cooperated with the author of CSPNet (A backbone for enhancing learning in terms of gradient) 


### Architecture wise change
- Replace FPN with Path Aggregation Network
{{< figure src="/ansonwang/images/PAN.png" caption="Fig1. (a) is FPN. (b) Bottom-up path augmentation. (c) Adaptive feature pooling. (d),(e) for prediction (src: Shu Liu et al, 2018)" >}}

- Introduce CSPNet as backbone

{{< figure src="/ansonwang/images/denseblock_gradient.png" caption="Fig2. Example of gradient flow of dense block. CSPNet bisects output on channel keep the rich of gradient paths and reduce BFLOPs." >}}



### Other tricks

Different tricks were experimented by YOLOV4, selected tricks can be categorized to these two classes:

- **Bag of Freebies:**
    Increase training cost but do not affect inference time, including loss function, regularization (DropBlock) methods, data augmentation(self-adversarial attack), class label smoothing.

- **Bag of Specials:**
    Slightly affact the inference time but greatly improve accuracy, SPP, Spatial Attention Module (SAM: max pooling+average pooling on channel and apply conv, sigmoid to generate an H x W matrix), Mish activation (Retains small negative values, allowing non-zero gradients when x < 0), Cross-stage partial connections (CSP), DIoU-NMS

**DIoU**

Takes location into account:

$$ \text{DIoU}(B_1, B_2) = \text{IoU}(B_1, B_2) - \frac{\rho^2(\mathbf{b}_1, \mathbf{b}_2)}{c^2} $$

$$ \text{IoU}(B_1, B_2) = \frac{|B_1 \cap B_2|}{|B_1 \cup B_2|} $$

$$ \rho^2(\mathbf{b}_1, \mathbf{b}_2) = (x_1 - x_2)^2 + (y_1 - y_2)^2 $$

$$ c^2 = (x_{\max} - x_{\min})^2 + (y_{\max} - y_{\min})^2 $$


### Grid Sensitive Decoder:

$$ \sigma $$ function is widely known for gradient issue near 1 and 0 which happens as the object cneter is near the grid boundary.

New decoder is applied for localization prediction.

$$ b_x = (1 + s_x) \sigma(t_x) - 0.5 s_x + c_x $$

$$ b_y = (1 + s_y) \sigma(t_y) - 0.5 s_y + c_y $$

$$ b_w = p_w e^{t_w}, b_h = p_h e^{t_h} $$ where $$s_{x,y}$$ give the scalibity max the prediction more flexible. 

**Edge case:** 
1. $$ \sigma(t_x) = 1 $$  $$ => b_x = 1 + 0.5s_x + c_x $$

2. $$ \sigma(t_x) = 0 $$
$$ => b_x = -0.5 s_x + c_x $$

3. $$ s_x = 0 $$
=> Degrade to the original version

## Scaled YOLOv4
YOLOv4 teams released scaled version in 2020 included tons of experiments for different scenarios. Summarized the contributions as follows:

-  Created a family of models (P5, P6, P7, Tiny), different sizes get different structure designs.
- Modular search in a predefined region, not black-box AutoML.
    - Backbones: CSPDarkNet, EfficientNet, MobileNet
    - Attention modules: SE, CBAM(Weights on channels), SAM (Weights on grids)
    - Necks: PANet, BiFPN
    - Head: Shallow, Deep, Shared, Seperate
    - Activation functions (Mish, Swish, ReLU)

- Treated the model as three modular blocks: [ Backbone ] → [ Neck ] → [ Head ], plug-and-play structure.


The main contirubtion of YOLOV5 from ultralytics is practicality. Skip the introduction of V5 here. 

## YOLOv6
Guessed V7 was named because Jocher Glenn added V6 on V5's repo. The other team submitted V6 after V7.

- Apply RepVGG on backbone
- Label Assignment: SimOTA and top-k selection (anchor free)
- Improvement tricks:
    - Cosine decay, label smoothing $$ y_{smooth} = y_{true}(1- \epsilon) + \epsilon/K $$, K is number of classes
    - EMA on weights for test time 
    - IoU loss (Direct prediction on IoU)
    - SyncBacthNorm (Mean/variance are synchronized across GPUs)
    - Self-distillation (Optional: EMA model provides soft label at the same time)
    - Quantization for depolyment (PTQ, QAT)

**SimOTA**

Simplified Optimal Transport Assignment
Restricts candidates to predictions whose center points fall within a fixed radius.
Dynamically selects positive matches based on a cost function, not just IoU. 


![Equation](https://latex.codecogs.com/png.latex?\dpi{120}\color{white}cost_{i,j}=\lambda_{cls}\cdot{Loss}^{cls}_{i,j}+\lambda_{IoU}\cdot(1-IoU_{i,j}))


and applied dynamic top-k selection. Here $$k=\sqrt{n}$$. This remove the needs of the anchors.


### Tricks for Inference

**RepVGG**

{{< figure src="/ansonwang/images/repvgg.png" caption="Fig3. RepVGG structure. It removed 1x1 and identity by re-parameterization tricks. Most of edges and optimizor were designed for 3x3 conv" >}}

$$ y = BN(Conv(x)) = \gamma \frac{Wx+b-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

can be rewritten as $$ y = W'x + b' $$ where 

$$ W' = \gamma \frac{W}{\sqrt{\sigma^2 + \epsilon}} $$

$$ b' = \gamma = \frac{b \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$


  
**Post Training Quantization**

Run on a small dataset to find min/max for each layer.

Scale: s = (max_float - min_float)/(max_int - min_int)
Zero_points: z = -min_float/scale

After quantization: q(x) = round(x/s) + z

**Quantization-Aware Training**

Simulate Quantization in Training or Finetuning(Most Time):

q(x)=round(x/s)⋅s 

Gradients are still computed in high precision (float32) during the backward pass to update weights accurately. (straight-through)

## YOLOv7
- E-ELAN: Divide the channel into groups -> Independant Conv on each group -> Go deeper 3-5 conv layers -> Channel Shuffling -> Merge (Repetition~~)
- Customized locations of RepVGG
- Design structure for each size, just like scaled YOLO-V4
- Auxiliary head training: Attaches one head on shallower layer to help early layers get better information.
- Label assignment: SimOTA uses one rule to start; YOLOv7 uses three, then does the same 
k-picking trick.

{{< figure src="/ansonwang/images/E-ELAN.png" caption="Fig4. Modified from ELAN (c), enchancing feature learning.(src: Chien-Yao Wangi, 2022)" >}}

{{< figure src="/ansonwang/images/Comparison_from_v6.png" caption="Fig5. YOLOv6 outperforms others in edge deployment latency vs AP (TensorRT + T4).(src: Chuyi Li, 2022)" >}}

{{< figure src="/ansonwang/images/Comparison_from_v7.png" caption="Fig6. YOLOv7 provides higher maximum AP (dataset: COCO test-dev) on V100 and strong overall performance.(src: Chien-Yao Wang, 2022)" >}}


## References
[1] Alexey Bochkovskiy, et al, YOLOv4: Optimal Speed and Accuracy of Object Detection, CVPR, 2020

[2] Gao Huang, et al, Densely Connected Convolutional Networks, CVPR, 2018

[3] Chien-Yao Wang, et al, Scaled-YOLOv4: Scaling Cross Stage Partial Network, CVPR, 2020

[4] Chien-Yao Wang, et al, CSPNet: A New Backbone that can Enhance Learning Capability of CNN, CVPR, 2019

[5] Tsung-Yi Lin, et al, Feature Pyramid Networks for Object Detection, CVPR, 2017

[6] Sanghyun Woo, et al, CBAM: Convolutional Block Attention Module, CVPR, 2018

[7] Shu Liu, et al, Path Aggregation Network for Instance Segmentation, CVPR, 2018

[8] Chuyi Li, et al, YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications, CVPR, 2022

[9] Chien-Yao Wang, et al, YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors, CVPR, 2022