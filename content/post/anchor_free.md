---
author: "Anson Wang"
date: "2025-12-14"
description: "CornerNet, CenterNet"
title: "CornerNet, CenterNet"
tags: [
    "Object Detection",
    "Anchor Free",
    "Computer Vision"
]
categories: ["Computer Vision"]
math: true
---



## CornerNet


### 1. The Problem: Limitations of Anchor-Based Methods

Traditional object detectors, whether two-stage (e.g., Faster R-CNN) or one-stage (e.g., SSD, RetinaNet), rely heavily on predefined **Anchor Boxes**. This methodology introduces two primary challenges:

* **Excessive Anchor Counts and Imbalanced Sampling:**
    * To ensure coverage of diverse object scales and aspect ratios, models must generate a massive number of anchors (e.g., RetinaNet uses up to $100k$).
    * The vast majority of these anchors are negative samples (background), leading to an extreme **positive-negative sample imbalance** during training, necessitating complex remedies like Focal Loss.

* **Over-reliance on Hyperparameters:**
    * The count, size, and aspect ratios of anchor boxes are delicate **hyperparameters** that require tedious manual tuning.
    * This dependency compromises the model's training efficiency and generalization ability, particularly when dealing with datasets containing objects of unusual aspect ratios.

CornerNet's core goal is to **completely eliminate the dependence on anchor boxes**, thereby simplifying the detection pipeline.

### 2. Core Idea: A Bottom-Up Paradigm Inspired by Pose Estimation

CornerNet transforms object detection into the problem of detecting and grouping **Corner Pairs**, a concept inspired by the **Bottom-Up** approach in **Multi-Person Pose Estimation**.

#### From Pose Estimation to Object Detection:

In Bottom-Up pose estimation, the process involves two steps: first, detecting all body keypoints, and second, using a grouping mechanism to assemble keypoints belonging to the same person. CornerNet adapts this:

1.  **Detecting Corner Pairs:** Instead of detecting the bounding box directly, the model locates the object's **Top-Left** and **Bottom-Right** vertices.
2.  **Grouping:** It then employs **Embedding Vectors** associated with each detected corner to determine if a top-left and a bottom-right corner belong to the same object, forming the final bounding box.

{{< figure src="/ansonwang/images/cornerstructure.png" >}}

#### Innovation I: Corner Pooling

A bounding box corner itself often lacks sufficient local features for accurate localization, as the critical features lie along the object boundaries (the top and left edges). To address this, CornerNet introduced the specialized **Corner Pooling** layer. 

* **Mechanism:** Corner Pooling aggregates features from the internal boundaries of the object towards the corner point.
* **Example (Top-Left Corner Pooling):** For a feature map point $(i, j)$, the operation aggregates features located to its **right** and **below**:
    1.  **Horizontal Aggregation:** Compute the cumulative maximum from the right boundary, moving **left** towards $j$.
    2.  **Vertical Aggregation:** Compute the cumulative maximum from the bottom boundary, moving **up** towards $i$.
    3.  The final result is the summation of these two aggregated features, ensuring the point $(i, j)$ captures rich boundary context from both the top and left sides of the potential object.

{{< figure src="/ansonwang/images/cornerpooling.png" >}}


### 3. Implementation and Architecture Drawbacks

#### Implementation Details:
{{< figure src="/ansonwang/images/cornerdetail.png" >}}

* **Backbone:** The **Hourglass Network**  is used as the backbone due to its strong ability to capture features across multiple scales, which is vital for precise point localization.
* **Key Components:**
    1.  **Detecting Corners:** Predicting two heatmaps (for top-left and bottom-right corners) trained using **Focal Loss**.
    2.  **Grouping Corners:** Predicting an **Embedding Vector** for each corner. The grouping is trained with a **Pull/Push Loss** (pulling same-object embeddings closer, pushing different-object embeddings apart).
    3.  **Offset Compensation:** An **Offset** vector is predicted to finely adjust the corner coordinates, compensating for the **quantization error** introduced by the downsampling process (e.g., factor of 4) in the network.

#### Drawbacks and Limitations

While innovative, the CornerNet architecture exhibits two primary limitations:

1.  **Grouping Errors (False Pairing):**
    * The pairing of corners is solely based on the proximity of their embedding vectors.
    * When the image contains **multiple similar and closely-packed objects** (e.g., a crowd, a stack of identical items), these objects' corners tend to produce **similar embedding vectors**.
    * This similarity frequently leads to **mispairing** or **missing pairing**, significantly degrading performance in **dense scene** detection tasks.

2.  **Lack of Center Information:**
    * The model defines an object using only two extreme points, neglecting the features and constraints of the object's **center or internal region**.
    * This makes the bounding box **regression less robust**, as slight inaccuracies in either corner can lead to significant shifts in the final box, resulting in less stable and potentially less accurate localizations compared to center-based methods.


---

## CenterNet: Architecture Enhancements

CenterNet (an improvement over CornerNet) is designed to resolve two major limitations of the original CornerNet: **Grouping Errors** and **lack of Robust Localization**.

### 1. Center Pooling and Center Constraint

Center Pooling is introduced primarily to directly address the issue of weak local features(lead to wrong bounding boxes) at an object's geometric center.

#### Core Motivation for Center Pooling

* **Weak Geometric Center Features:** The geometric centers of objects do not necessarily convey very recognizable visual patterns (e.g., the human head contains strong visual patterns, but the center keypoint is often in the middle of the human body).
* **Solution:** Center Pooling is designed to force the potential center point to aggregate features from the object's four boundaries, thereby indirectly verifying its central position and serving as a powerful **geometric filter** to actively resolve grouping errors.

#### Center Pooling Mechanism

{{< figure src="/ansonwang/images/centerpooling.png" >}}

The Center Pooling module performs accumulated maximum pooling in both **horizontal and vertical directions**. This ensures that only a point truly located at the object's center can simultaneously perceive strong feature information from the object's four boundaries (top left, bottom right).

> ** Potential Center Pooling Issue:** Due to its four-directional accumulated maximum design, Center Pooling can be susceptible to **cross-object interference**. For instance, when a small object is near a large one, the small object's center may erroneously aggregate high-intensity features from the larger object's boundaries, potentially inflating the confidence of the small object's center and increasing the risk of false positives.

### 2. Cascade Corner Pooling (CCP) and Enhanced Localization

CCP's main goal is to improve the accuracy of corner localization, which in turn contributes indirectly to more stable corner pairings and reduced grouping errors.

#### Core Motivation for CCP

* **Defect of Original CP:** Corners are often outside the objects, which lacks local appearance features.
* **CP Search Limitation:** Original Corner Pooling tends to aggregate features from the entire boundary (globally), making it susceptible to long-distance features and leading to a **lack of precise localization ability for an object's local boundary features**.


{{< figure src="/ansonwang/images/cascadeexplaination.png" >}}

Path is from '1' to '2', the pink region is not included in pooling op.

#### CCP Mechanism: Two-Stage Search for Localization Enhancement 

CCP employs a two-stage "cascaded" process, using a segmented search to better leverage localization information:

1.  **Stage One ($V_b$):** Find the point $P_b$ with the strongest response along the boundary line (the boundary maximum value).
2.  **Stage Two ($V_i$):** Starting from the locked **local position $P_b$**, search along the **object's internal direction** (e.g., vertically downward) for the internal feature $V_i$ (the internal maximum value).

* **Effect:** By segmenting the search, CCP successfully combines the global boundary information ($V_b$) with the **local internal features** ($V_i$) corresponding to the strongest point on that boundary. This significantly improves corner localization accuracy and **indirectly helps to mitigate grouping errors** by providing more reliable corner points.

---


## References
[1] Duan, Kaiwen, et al. "Centernet: Keypoint triplets for object detection." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

[2] Law, Hei, and Jia Deng. "Cornernet: Detecting objects as paired keypoints." Proceedings of the European conference on computer vision (ECCV). 2018.
