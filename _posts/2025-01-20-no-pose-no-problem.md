---
layout: post
title: Simple 3D Gaussian Splats from Unposed Images
date: 2025-01-20
description: NoPoSplat introduces a feed-forward network that directly predicts 3D Gaussian parameters within a canonical space from sparse, unposed multi-view images.
tags: gaussian-splatting
categories: paper-notes
thumbnail: assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_21.20.30.png
disqus_comments: true
---

- [NoPoSplat Project Page](https://noposplat.github.io/)
- [Summary version with cited in Notion](https://quan283.notion.site/No-Pose-No-Problem-Surprisingly-Simple-3D-Gaussian-Splats-from-Sparse-Unposed-Images-1832a3f8f2a180ec8690d5080a39394f?pvs=4)

# 1. Introduction and Problem Statement

Existing state-of-the-art (SOTA) methods for generalizable 3D scene reconstruction, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), require **accurate camera poses** as input. These poses are typically obtained through **Structure-from-Motion (SfM)** methods like COLMAP. However, this requirement is impractical in many real-world scenarios, especially when dealing with **sparse views** (e.g., just a couple of images) or in **textureless areas** where pose estimation is unreliable.

## Limitations of Existing Methods

Current methods that attempt to jointly estimate poses and reconstruct scenes suffer from a **compounding effect**: errors in initial pose estimates degrade reconstruction quality, which in turn further degrades pose accuracy. This creates a feedback loop that limits the effectiveness of these approaches.

## NoPoSplat's Solution

NoPoSplat introduces a **feed-forward network** that directly predicts 3D Gaussian parameters within a **canonical space** from sparse, unposed multi-view images. This eliminates the need for explicit pose estimation during reconstruction. By anchoring the first input view's local camera coordinates as the canonical space, NoPoSplat avoids the need to transform Gaussians from local to global coordinates, thus bypassing the errors associated with pose estimation.

# 2. Related Works

## 2.1. Generalizable 3D Reconstruction and View Synthesis

NeRF (Neural Radiance Fields) and 3DGS (3D Gaussian Splatting) have significantly advanced 3D reconstruction and novel view synthesis, but these methods typically require **dense posed images** as input and **per-scene optimization** limits their practical application

Recent approaches focus on **generalizable 3D reconstruction** from sparse inputs**.** These methods typically use task-specific backbones that leverage **geometric information** to enhance scene reconstruction. Examples include:

- **MVSNeRF and MuRF**: These methods build **cost volumes** to aggregate multi-view information
- **pixelSplat**: This method employs **epipolar geometry** for improved depth estimation

However, these geometric operations often require **camera pose input** and sufficient **camera pose overlap** among input views. In contrast, NoPoSplat uses a **Vision Transformer (ViT)** without any geometric priors, making it **pose-free** and more effective in handling scenes with **large camera baselines**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_21.20.57.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## 2.2. Pose-Free 3D Scene Reconstruction

Classical methods rely on accurate camera poses of input images, typically obtained through Structure from Motion (SfM) methods like COLMAP, which complicates the overall process.

Some recent works attempt to **jointly optimize camera poses** and neural scene representations, but they still require rough pose initialization or are limited to small motions. Other methods adopt **incremental approaches**, but they only allow image/video sequences as input

For generalizable sparse-view methods, requiring camera poses during inference presents significant challenges, as these poses are often unavailable in real-world applications during testing. While two-view pose estimation methods can be used, they are prone to failure in textureless regions or when images lack sufficient overlap. Some recent pose-free novel view synthesis methods break the task into two stages: first estimate camera poses, then construct the scene representation. However, this two-stage process lags behind pose-required methods due to noise from the initial pose estimation.

In contrast, NoPoSplat completely eliminates camera poses by directly predicting 3D Gaussians in **a canonical space**, avoiding potential noise in pose estimation and achieving better scene reconstruction

**Splatt3R**, a concurrent work, also predicts Gaussians in a global coordinate system but relies on the frozen **MASt3R model** for Gaussian centers and requires ground truth depth during training, which makes it unsuitable for novel view synthesis and limits its ability to leverage widely available video data

# 3. Methodology

## 3.1. Problem Formulation:

The goal is to reconstruct a 3D scene from a set of unposed multi-view images and their corresponding camera intrinsics, represented as $\{I^v, k^v\}$

Here, $I^v$ is the input image for view $v$, and $k^v$   represents the camera intrinsic parameters for that view.

The network, denoted as $f_θ$ with learnable parameters $θ$, maps these inputs to 3D Gaussians within a **canonical 3D space**. The mapping is defined as:

$$
fθ : \{(I^v,k^v)\}^V_{v=1} → {∪ (\mu^v_j ,\alpha^v_j ,r^v_j , s^v_j , c^v_j )}^{v=1,...,V}_{j=1,...,H \times W}
$$

- $\mu$:  the center position of the Gaussian primitive in $R^3$
- $\alpha$:  the opacity of the Gaussian primitive in $R$.
- $r$:  the rotation factor in quaternion in $R^4$
- $s$:  the scale of the Gaussian primitive in $R^3$
- $c$:  the spherical harmonics (SH) coefficients of the Gaussian primitive in $R^k$

The method assumes that camera intrinsics (*k*) are available, which is a common assumption as they are generally available from modern devices.

The method can generalize without any optimization. The results 3D Gaussians in canonical space enabble two tasks

1. **Novel view synthesis**: Given a target camera transformation relative to the first input view, the model renders a novel view.
2. **Relative pose estimation**: The model estimates the relative camera poses between input views.

## 3.2. Overall Pipeline

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_21.20.30.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The NoPoSplat pipeline consists of three main components:

1. **Encoder**: A Vision Transformer (ViT) that processes the input images and camera intrinsics.
2. **Decoder**: Another ViT that integrates multi-view information through cross-attention.
3. **Gaussian Parameter Prediction Heads**: Two prediction heads that output the Gaussian parameters (center positions and other attributes).

Both the encoder and decoder use Vision Transformer (ViT) architectures without any geometric priors, which is a key difference from other methods that rely on geometric priors like epipolar constraints or cost volumes. The use of a pure ViT structure allows the model to be more flexible and effective when the overlap between input views is limited.

### 3.2.1 Vision Transformer (ViT) Encoder and Decoder

The RGB images are converted into sequences of image tokens, and these are concatenated with an **intrinsic token** (Detail in Section 3.4). These tokens are fed into a ViT encoder separately for each view. The encoder shares weights across all input views.

The output features of the encoder are passed to a ViT decoder, which integrates multi-view information through **cross-attention layers**, allowing the features from each view to interact with those from all other views.

### 3.2.2 Gaussian Parameter Prediction Heads

[DPT](https://huggingface.co/docs/transformers/main/en/model_doc/dpt)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/image.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
The DPT model was proposed in Vision Transformers for Dense Prediction by René Ranftl, Alexey Bochkovskiy, Vladlen Koltun. DPT is a model that leverages the Vision Transformer (ViT) as backbone for dense prediction tasks like semantic segmentation and depth estimation.
</div>

Two prediction heads are used to predict the Gaussian parameters, based on the **DPT architecture.**

- The first head focuses on predicting **Gaussian center positions ($µ$)** and takes features exclusively from the transformer decoder.
- The second head predicts the other Gaussian parameters (α, r, s, c) and uses both the ViT decoder features and an RGB image shortcut as input.

The RGB image shortcut allows for the direct flow of texture information which is crucial for capturing fine details. The ViT decoder features are downsampled, so they lack detailed structural information which the RGB shortcut compensates for

## 3.3. Canonical Gaussian Space

Instead of predicting Gaussians in each local camera coordinate system and then transforming them to a world coordinate system using camera poses, NoPoSplat directly outputs Gaussians in a canonical space (Figure 5 and 6)

**The local camera coordinate of the first input view is anchored as the canonical space**, meaning that the camera pose for the first view is considered to be [U  0], where $U$ is a unit/identity matrix for rotation and 0 is a zero translation vector.

All Gaussians are predicted relative to this canonical space. The network predicts the set

$$
\{µ^{v→1}_j, r^{v→1}_j, c^{v→1}_j, α_j, s_j\}
$$

where the superscript $v→1$ denotes the Gaussian parameters corresponding to pixel $p_j$ in view $v$, under the local camera coordinate system of view 1.

This approach eliminates the need for camera poses and allows the network to learn the fusion of different views directly within the canonical space, which results in a more cohesive global representation

## 3.4. Camera Intrinsics Embedding
The camera intrinsics ($k = [f_x, f_y, c_x, c_y]$) are crucial for resolving the scale ambiguity

Three methods for embedding camera intrinsics are compared

1. **Global Intrinsic Embedding - Addition:** Camera intrinsics are fed into a linear layer to obtain a global feature, which is added to the RGB image features after the patch embedding.
2. **Global Intrinsic Embedding - Concatenation:** The global feature is treated as an intrinsic token and concatenated with the image tokens.
3. **Dense Intrinsic Embedding**: Per-pixel camera rays are converted to higher-dimension features using spherical harmonics and are concatenated with the RGB image.

The **intrinsic token method (concatenation) performs the best.** This method is not only effective at injecting camera intrinsic information but also gives the best performance.

## 3.5 Training and Inference

### 3.5.1 Training

The 3D Gaussians are used to render images at novel viewpoints, and the network is trained end-to-end using **ground truth target RGB images** as supervision. The loss function is a linear combination of **MSE (Mean Squared Error)** and **LPIPS (Learned Perceptual Image Patch Similarity)** loss with weights 1 and 0.05, respectively.

### 3.5.2 Relative Pose Estimation

The 3D Gaussians in the canonical space are directly used for relative pose estimation. The pose estimation is approached in two steps:

1. **Coarse Pose Estimation**: The an initial pose estimate is obtained using the **PnP (Perspective-n-Point) algorithm** with **RANSAC**, given the Gaussian centers.
2. Refinement: The initial pose is refined by rendering the scene using the estimated pose and optimizing the alignment with the input view using **photometric loss** and the structural part of the **SSIM loss**.

The **camera Jacobian** is calculated to reduce computational overhead during optimization.

### 3.5.3 Evaluation Time Pose Alignment

> “”However, 3D scene reconstruction with just two input views is inherently ambiguous as many different scenes can produce the same two images. As a result, though the scene generated by our method successfully explains the input views, it might not be exactly the same as the ground truth scene in the validation dataset.””
>

For evaluation purposes, the camera pose for the target view is optimized to match the ground truth image by freezing the Gaussians and optimizing the target camera pose using photometric loss. This is done to compare with other baselines, but is not required for real world use (Figure 7)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_22.28.49.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

# 4. Experiment Results and Analysis

The experiments use the **RealEstate10k (RE10K)** and **ACID** datasets, and also test **zero-shot generalization** on **DTU** and **ScanNet++** datasets. To handle input images with varying camera overlaps, they generate input pairs for evaluation that are categorized based on the ratio of image overlap

The evaluation metrics used include **PSNR**, **SSIM**, and **LPIPS** for novel view synthesis, and **AUC (Area Under the Curve)** for pose estimation.

## 4.1. Novel View Synthesis

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_22.38.47.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Table 1: Novel view synthesis performance comparison on the RealEstate10k (Zhou et al., 2018) dataset. Our method largely outperforms previous pose-free methods on all overlap settings, and even outperforms SOTA pose-required methods, especially when the overlap is small.
</div>

**NoPoSplat significantly outperforms all state-of-the-art pose-free methods in novel view synthesis.** Methods such as DUSt3R and MASt3R struggle to fuse input views effectively due to their reliance on per-pixel depth loss, and Splatt3R inherits this limitation

NoPoSplat achieves **competitive performance with state-of-the-art pose-required methods** like pixelSplat and MVSplat, and **outperforms them particularly when the overlap between input images is small.**

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_22.46.00.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Table 2: Novel view synthesis performance comparison on the ACID (Liu et al., 2021) dataset.
</div>

This improved performance is attributed to NoPoSplat's 3D Gaussian prediction in a canonical space, as opposed to the transform-then-fuse strategy used by other methods.

Qualitative results in Figure 4 demonstrate NoPoSplat's ability to achieve more coherent fusion from input views, superior reconstruction with limited image overlap, and enhanced geometry reconstruction in non-overlapping regions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_22.46.00.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Figure 4: Qualitative comparison on RE10K (top three rows) and ACID (bottom row). Compared to baselines, we obtain: 1) more coherent fusion from input views, 2) superior reconstruction from limited image overlap, 3) enhanced geometry reconstruction in non-overlapping regions
</div>

## 4.2. Relative Pose Estimation

NoPoSplat shows strong performance in **relative pose estimation** on the **RE10k**, **ACID**, and **ScanNet-1500** datasets. The method is trained on either **RE10K** (denoted as Ours) or a combination of **RE10K** and **DL3DV** (denoted as Ours*).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_22.46.53.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Table 3 provides the pose estimation performance using the AUC metric with thresholds of 5°, 10°, and 20° across the different datasets
</div>

The results demonstrate that performance consistently improves when scaling up training with **DL3DV**. **NoPoSplat demonstrates superior zero-shot performance on ACID and ScanNet-1500**, even outperforming **RoMa**, which was trained on **ScanNet**. The authors state that this indicates the effectiveness of the pose estimation approach and the quality of the 3D geometry produced.

## 4.3. Geometry Reconstruction

NoPoSplat outputs improved 3D Gaussians and depths when compared to state-of-the-art pose-required methods.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_21.53.26.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Figure 5 shows that MVSplat suffers from misalignment in the intersection regions of input images, and distortions in areas with limited overlap. These problems are attributed to the transform-then-fuse pipeline.
</div>

Figure 5 shows that MVSplat suffers from misalignment in the intersection regions of input images, and distortions in areas with limited overlap. These problems are attributed to the transform-then-fuse pipeline**8**.

NoPoSplat's direct Gaussian prediction in canonical space addresses these issues. The results show that even without camera poses as input, NoPoSplat generates higher quality 3D Gaussians, resulting in improved color and depth rendering

## 4.4. Cross-Dataset Generalization

**NoPoSplat exhibits superior zero-shot performance on out-of-distribution data**, when trained exclusively on RealEstate10k and tested on ScanNet++ and DTU datasets. The model's minimal geometric priors help it to adapt effectively to different scene types.

As shown in Table 4 and Figure 6, NoPoSplat outperforms state-of-the-art pose-required methods on out-of-distribution data.

Notably, NoPoSplat outperforms Splatt3R, even on ScanNet++ where Splatt3R was trained.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_21.53.59.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Figure 6: Cross-dataset generalization. Our model can better zero-shot transfer to out-ofdistribution data than SOTA pose-required methods.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_22.58.05.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Table 4: Out-of-distribution performance comparison. Our method shows superior performance when zero-shot evaluation on DTU and ScanNet++ using the model solely trained on RE10k.
</div>


## 4.5. Model Efficiency

NoPoSplat can predict 3D Gaussians from two **256x256** input images in **0.015 seconds (66 fps)** on an **RTX 4090 GPU**. This speed is approximately **5 times faster** than **pixelSplat** and **2 times faster** than **MVSplat**, which shows the benefits of using a standard ViT without additional geometric operations.

## 4.6. Real Application

NoPoSplat can be directly applied to **in-the-wild unposed images**, including images taken with mobile phones and frames extracted from videos generated by **Sora**. The results indicate potential applications of NoPoSplat in **text/image to 3D scene generation** pipelines.

Figure 7 shows results from in-the-wild data, demonstrating the method's applicability to sparse image pairs from mobile phones and frames from Sora-generated videos.

# 5. Abalation Studies
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_23.03.32.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Table 5: Ablations. intrinsic embeddings are vital for performance and using intrinsic tokens performs the best. Adding the RGB image shortcut also improves the quality of rendered images. Our method achieves better performance compared with the poserequired per-local-view Gaussian field prediction method.

</div>



## 5.1. Ablation on Output Gaussian Space

The authors compare NoPoSplat's canonical Gaussian space prediction with the transform-then-fuse pipeline used by pose-required methods.

The transform-then-fuse method predicts Gaussians in each local camera coordinate system and then transforms them to a world coordinate system using camera poses. For a fair comparison, both methods use the same backbone and head, but differ in the prediction of Gaussian space.

Results in row (f) of Table 5 show that the **pose-free canonical space prediction** method outperforms the transform-then-fuse approach. Figure 8 shows that the transform-then-fuse strategy leads to ghosting artifacts in the rendering due to difficulties in aligning Gaussians from different input views.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/no_pose_no_problem/Screenshot_2025-01-22_at_23.03.55.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Figure 8: Ablations. No intrinsic results in blurriness due to scale misalignment. Without the RGB image shortcut, the rendered images are blurry in the texture-rich areas. Using the transform-then-fuse strategy causes ghosting problem.
</div>

## 5.2. Ablation on Camera Intrinsic Embedding

Three intrinsic encoding strategies are studied, along with a scenario where no intrinsic information is provided. The results in Table 5 row (b) and Figure 8 show that **not using intrinsic encodings leads to blurry results due to scale ambiguity.**

The intrinsic token embedding consistently performs the best among the three proposed methods, and is used as the default setting. The three methods compared are (b) (c) (d)

## 5.3. Importance of RGB Shortcut

The impact of using an RGB image shortcut to the Gaussian parameter prediction head is studied. In addition to low-resolution ViT features, RGB images are input into the Gaussian parameter prediction head.

Figure 8 shows that **without the RGB shortcut, rendered images are blurry in texture-rich areas**

## 5.4. Extend to 3 Input Views

The method is extended to use three input views to show how additional views improve performance. The third view is added between the two original input views.

Row (g) of Table 5 shows that **performance significantly improves with the inclusion of an additional view**

# 6. Closing

**Reliance on Known Camera Intrinsics:** NoPoSplat, like other pose-free methods, assumes that camera intrinsics are known. Although the authors note that heuristically set intrinsic parameters work well for in-the-wild images, they acknowledge that relaxing this requirement would improve the method's robustness in real-world applications.

**Non-Generative Model:** As a feedforward model, NoPoSplat is not generative. This means it **cannot reconstruct unseen regions of a scene with detailed geometry and texture.** This limitation is apparent in the 2-view model, where areas not covered by the input views may not be accurately reconstructed. The authors suggest that incorporating additional input views could potentially mitigate this issue.

**Limited Training Data:** The current training data is limited to the RealEstate10K, ACID, and DL3DV datasets. This **constrains the model's ability to generalize to diverse in-the-wild scenarios**. The authors propose training the model on larger, more diverse datasets in the future to address this.

**Static Scenes:** The method is currently limited to static scenes.

