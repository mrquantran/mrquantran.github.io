---
layout: post
title: 3D Gaussian Splatting
date: 2024-08-15
description: 3D Gaussian splatting is a technique used in the field of real-time radiance field rendering.
tags: gaussian-splatting
categories: paper-notes
thumbnail: assets/img/paper-notes/gaussian_splatting/ejemplo_gaussian_splatting.png
---

The paper begins by highlighting the recent revolution in novel-view synthesis using `Neural Radiance Field` methods, which renders 3D scenes from various viewpoints, given multiple images and their corresponding camera pose values

However, it points out two key problems: achieving high visual quality requires computationally expensive neural networks for training and rendering, and faster methods often sacrifice visual quality for speed. The problem is further compounded when rendering unbounded scenes at 1080p resolution, where no existing method can achieve real-time display rates.

This paper has become a hot topic in the `novel-view synthesis field`. It surpasses `Mip-NeRF360` (2022), the SOTA (state-of-the-art) method in rendering quality for high-resolution (1920x1080) outputs, and even reduces training time compared to `InstantNGP` (2022), the SOTA method for speed.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/compared-nerf.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

In the diagram above, the values in parentheses represent `rendering speed (fps)`. It shows how much improvement has been achieved compared to existing SOTA studies. The prospect of real-time services using NeRF has come one step closer. (Although there was previously a method called `FastNeRF`, capable of rendering at 200 FPS, it had slow training times and lower quality.)

> The reason this paper has gained significant attention is its revolutionary fast rendering speed (exceeding approximately 100 FPS), in addition to the advantages mentioned above.

<div class="row mt-3 mb-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/results_evalute.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

# Why use 3D Gaussian ?
To introduce two methods of representing 3D models:
1. The most common methods are **Meshes** and **Points**. These are optimized for **rasterization** (the process of converting 3D into 2D images) based on GPU/CUDA.
2. In recent NeRF techniques, scenes are represented as **continuous scenes**, which are well-suited for optimization. However, during rendering, the **stochastic (probability-based) sampling** process requires significant computation and may produce noise.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/Dolphin_triangle_mesh.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/nerf_repersentation.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
Let's explain this from a rendering perspective from existing NeRF:
1. Draws a ray for each pixel of the image and samples several points along the ray.
2. Calculates the color and volume density for each sampled point.
3. Summates these values along the ray to render the image.

This paper combines the advantages of the aforementioned 3D model representation methods and proposes a novel approach called **3D Gaussian**.
- Supports **differentiable volumetric representation**,
- Allows **explicit representation** (unlike neural networks, which use implicit representations with hidden structures),
- Efficiently performs **2D projection** of 3D models and **α-blending** (an additive operation for transparency values), enabling **fast rendering**.

> Note: 2D projection and α-blending are parts of the rasterization process

From a computational perspective, the rendering operations required for 3D Gaussian Splatting are significantly lower compared to conventional NeRF.

# Overview
Let’s first take a look at the overall pipeline. Look at it to get understanding the big picture.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/overview_pipeline.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

1. **Initialization**: Using an **SfM (Structure-from-Motion)** algorithm like COLMAP, not only the camera poses but also the point cloud data can be obtained. These point clouds are used as the initial values for the 3D Gaussians.
2. **Projection**: The 3D Gaussians are projected onto the image plane (a plane at a distance of 1 unit along the z-axis from the camera) to form 2D Gaussians. This step is performed to *update parameters by comparing with ground truth (GT) input images*.
3. **Differentiable Tile Rasterizer**: A differentiable tile rasterization process generates the 2D Gaussians into a single image.
4. **Gradient Flow**: The generated image and the GT image are used to *calculate the loss*, and gradients are backpropagated accordingly.
5. **Adaptive Density Control**: Based on the gradients, the shape of the Gaussians is adjusted.

This paper review will be explained based on pseudocode. While it might seem rigid, this approach is chosen as it is the most effective way to explain unfamiliar concepts. The actual paper also feels like reading a written manual.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/algorithm_1.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The overall process can be divided into three main parts:
1. **The first part (red)**: This is the variable initialization step. Although it may appear simple from a coding perspective, it is critical as it pertains to model design.
2. **The second part (blue)**: This structure is familiar in the ML field. It involves taking inputs, performing inference, calculating the loss, and updating parameters.
3. **The third part (green)**: This part directly manipulates the Gaussians. At specific iterations, Gaussians are cloned, split, and removed.

# Initialization
M, S, C, and A are the learning parameters. Below is the explanation of the meaning and initial values of each parameter:
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/initialization.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

$M$ (mean): represents the point cloud obtained using SfM (Structure-from-Motion). A 3D Gaussian is composed of a mean and a covariance. The points in the point cloud are used as the initial mean values of the 3D Gaussians. As a result, the number of 3D Gaussians generated is equal to the number of points in the point cloud.

$S$ (Covariance Matrix): is the covariance matrix of the 3D Gaussian, which is a $3 \times 3$ matrix. In the mathematical formulation in the paper, it is described as $\Sigma$, composed of a Scale Matrix (S) and a Rotation Matrix (R). (Note: the notation differs between the pseudocode and the equations in the paper, as shown below.)

\begin{equation}\Sigma = R SS^TR^T\end{equation}
- The design separates the scaling factor $s$ and the quaternion $q$ (one of the ways used for rotation representation) into two independent factors for optimization.
- The scaling vector $s$ contains information about scaling along the $x$, $y$, and $z$ axes.
- The quaternion representation $q$ (shape: $4 \times 1$) is converted into a rotation matrix $R$ (shape: $3 \times 3$)
- The reason for this design is to ensure that the covariance matrix is positive definite when projecting the 3D Gaussian into a 2D Gaussian for rendering. A positive definite matrix ensures that all variables have values greater than 0.

$C$ (color) represents the color value of the 3D Gaussian, and color is designed using the Spherical Harmonics (SH) function. Each 3D Gaussian is designed to find the optimal SH coefficient according to the view direction

$A$ (transparency) represents the transparency (alpha) value of the 3D Gaussian and is a single scalar value.

# Differentiable 3D Gaussian Splatting
The 3D Gaussians are projected to 2D splats for rendering. . This projection is done using a viewing transformation, and a 2x2 covariance matrix $\sigma$ is obtained that has similar properties as if the method started from planar points with normals.
\begin{equation}\Sigma' = JW\Sigma W^T J^T\end{equation}
- **Projective Transformation**: The Jacobian matrix $J$ (a matrix of partial derivatives) for converting from the camera coordinate system to the image coordinate system.
- **Viewing Transformation**: The transformation matrix $W$ for converting from the world coordinate system to the camera coordinate system.
- **Covariance Matrix**: $\Sigma$, the covariance matrix in the world coordinate system.


# Optimization

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/optimization.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

From this point, we delve into the operations executed within the **train loop**.

**Line 1**: Simply reads the target image $\hat{I}$ and its corresponding camera pose information $V$.

**Line 2**: The inputs— $M$(mean = $xyz$), $S$ (covariance), $C$(color), $A$ (transparency), and $V$ (camera pose)—are passed to the Rasterizer, which generates the predicted image.

**Line 3**: The predicted image is compared with the ground truth (GT) image to compute the **loss**. The loss function is designed as a combination of **L1 loss** and **D-SSIM**, with $\lambda = 0.2$. Additionally, for $M$(mean = $xyz$), the **standard exponential decay scheduling** is applied, similar to the Plenoxel approach.

\begin{equation}L = (1-\lambda) L_1 + \lambda L_{DSSIM}\end{equation}

**Line 4**: The $M$, $S$, $C$, and $A$ values are updated using the Adam optimizer.

# Rasterizer
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/rasterizer.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The algorithm takes as input:
- Image dimensions (w, h),
- Gaussian means and covariances in world space (μ, Σ),
- Gaussian colors and opacities (c, α),
- Camera pose for the image (V).

## Cull Gaussian:
The view frustum is a 3D volume representing the visible area of the camera, and the culling process determines which 3D Gaussians fall within this volume.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/view_frustum.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

This step performs frustum culling to remove 3D Gaussians that are not within the current camera's view

The algorithm uses a 99% confidence interval of each Gaussian, keeping only the Gaussians that intersect the view frustum, and using a guard band to reject those at extreme positions. This avoids unnecessary computation on invisible or problematic Gaussians

## Screen Space Gaussian
Transforms the 3D Gaussian parameters from world space to screen space, preparing them for 2D rasterization.

The 3D Gaussian means (μ) and covariances (Σ) are transformed into camera coordinates using a viewing transformation (V) and the Jacobian of the affine approximation of the projective transformation. The result, $M'$ and $S'$, represent the means and covariances in screen space. The original 3D covariance matrix (Σ) is reduced to a 2x2 covariance matrix during the projection.

This step involves matrix multiplication and affine transformations to change coordinate systems, specifically projectively transforming the 3D Gaussians into a 2D view space.

## Create Tiles
This step divides the image space into smaller regions to enable parallel processing of different screen region. Creates a grid of tiles over the image, splitting the screen into 16x16 pixel tiles for parallel processing

This is a straightforward partitioning of the screen into equally sized rectangular tiles.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/gaussian_view_frustum.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/pixel_tiles.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Duplicate With Keys
Each Gaussian in the scene ($M'$, $S'$) is duplicated for each tile it overlaps. A key is generated for each instance. This key combines the **view space depth** of the Gaussian and the **tile ID**. The key is constructed so that the lower bits represent the projected depth and the higher bits the tile ID.

The resulting list contains the Gaussian instances and their corresponding keys (L, keys).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/duplicate_with_keys.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>


## Sort by Keys
Sorts the Gaussian instances based on the generated keys for each tile.

A fast GPU Radix Sort is used to sort the Gaussian instances (L) using the keys (keys), ensuring the Gaussian instances are ordered by depth within each tile. The Radix sort is a non-comparative integer sorting algorithm that sorts data with integer keys by grouping the keys by individual digits that share the same significant position and value. It provides a fast, parallel way of sorting.

This sorting is done once for each frame and is a global sort across all tiles.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/sort_by_keys.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Identify - Read Tile Ranges
Creates per-tile lists of Gaussians, by identifying start and end indices within the sorted keys array. By comparing neighboring elements of the sorted keys array, it identifies the start and end indices for each tile. The result, R, is a set of per-tile lists of Gaussians.

Read the range r for each tile in the image.

## Blend In Order
Blends the colors and opacities of the sorted Gaussians for each pixel within the tile to create the final pixel color. The Gaussians in the tile’s range, as defined in range, are blended front-to-back. The color and opacity of each Gaussian that overlaps the current pixel are combined with existing pixel color and opacity using alpha-blending. If the accumulated opacity of the pixel is close to 1, the loop is terminated early.

The contribution of a Gaussian is determined by a 2D Gaussian function with covariance $Σ'$, and a learned per-point opacity.

During the rasterization process, the algorithm also ensures numerical stability by
- Skipping blending updates with alpha values lower than a small threshold.
- Clamping the accumulated alpha values.
- Stopping the blending operation early if the accumulated opacity of the pixel exceeds a threshold.

# Adaptive Control of Gaussians
This is the stage where 3D Gaussians are adaptively modified to suit the scene. While the previously mentioned parameters $M, S, C, A$ are updated in each iteration, the processes in the green section are executed every 100 iterations. In this stage, 3D Gaussians undergo removal, splitting, or cloning, collectively referred to as densification.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/adaptive_density_control.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/figure_adaptive_control.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Remove Gaussian
Gaussians with an alpha value ($\alpha$, transparency) below a specific threshold ($\varepsilon$) are removed. In the code, this threshold is set to 0.005

After removing Gaussians step, the system addresses regions where geometric features are **not well-captured** (=Under-reconstruction) and regions where Gaussians have modeled overly broad areas (=Over-reconstruction). Specifically:
- Both under-reconstruction and over-reconstruction regions are identified as having a **high view-space positional gradient**.
- If the **average magnitude** of the view-space positional gradient exceeds a certain **threshold** (set at **0.0002**), the Gaussians in these regions are subjected to **cloning** or **splitting**.

## Clone Gaussian
For **under-reconstructed regions**:
- **Small Gaussians** (Gaussians with **low covariance**) are **duplicated**.
- The cloned Gaussians are placed **along the direction of the positional gradient**.

## Split Gaussian
For **over-reconstructed regions**:
- **Large Gaussians** (Gaussians with **high covariance**) are **split into smaller Gaussians**.
- A single Gaussian is divided into **two Gaussians**, and the scale is reduced by a factor of **1.6** (determined experimentally).
- The positions of the split Gaussians are assigned based on the **probability density function** of the original Gaussian.

## Optimization
In the case of **cloning**, both the number of Gaussians and the scene’s volume increase. On the other hand, with **splitting**, the total volume is maintained while the number of Gaussians increases. As with other volumetric techniques, **floaters** appear in regions closer to the camera, manifesting as randomly scattered Gaussians.

To address this, a strategy is employed where the **alpha values are reset to 0 every 3000 iterations**. During the stage where $M$, $S$, $C$, and $A$ are optimized, the alpha values transition from 0 to non-zero values over **100 iterations**. After these 100 iterations, the unwanted values are removed through the **Remove Gaussian** operation during the **Gaussian densification** phase.

Another benefit of this approach is that it also addresses cases where 3D Gaussians overlap. By periodically resetting the alpha values to 0, instances of large Gaussians overlapping are effectively eliminated.

The strategy of periodically resetting alpha values to 0 plays a significant role in controlling the overall number of Gaussians.

# Evaluation
Mip-NeRF360 utilized **4 A100 GPUs**, whereas the others used **A6000 GPUs**. FPS refers to the rendering speed. Compared to Instant-NGP, **3D Gaussian Splatting** achieves a similar training speed but higher **PSNR** and, most importantly, **significantly faster rendering speed**.

From the perspective of rendering speed, Instant-NGP’s **~10 FPS** might seem acceptable, but this speed was achieved using high-spec GPUs. If run on low-cost GPUs, **stuttering** could occur.

Drawbacks:
- Unlike previous methods, this approach consumes **a significant amount of memory**.
- During training for very large scenes, **GPU memory usage reached up to 20GB**.
- The source code explicitly recommends using GPUs with at least **24GB of VRAM**.
- For smaller scenes, GPUs with less memory can be used.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/quality.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

# Ablation Study
## Initialization from SfM
This section addresses experiments on the initialization of 3D Gaussians using the **SfM point cloud**. When creating a cube three times the size of the input camera's bounding box and sampling it uniformly, it was found that the method performed relatively well without completely failing, even in the absence of SfM points. As shown in the figure below, performance degradation is mainly observed in the background.

In areas not covered by the training views, initializing with random values results in floaters that cannot be eliminated through optimization. From another perspective, synthetic NeRF datasets do not exhibit this issue because they lack backgrounds, and the input camera pose values are well-defined.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/ablation_sfm.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Densification
They evaluated the densification method from the perspectives of cloning and splitting. Each method was disabled individually, while the others were optimized without modification. As seen in the figure below, splitting large Gaussians helps reconstruct the background effectively, whereas cloning small Gaussians enables faster convergence.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/ablation_densification.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Limitations
Artifacts can occur in sparse scenes with insufficient input images. While anisotropic Gaussians offer many advantages, they can sometimes result in elongated artifacts or blotchy Gaussians. (This phenomenon has also been observed in other studies.) When creating large Gaussians through optimization, artifacts occasionally appear, primarily in areas that exhibit different appearances depending on the viewing pose.
- One reason for these artifacts is the trivial rejection of Gaussians during the rasterizer stage through a guard band. Using a more theoretical culling approach could mitigate these artifacts.
- Another reason lies in the simple visibility algorithm, which can lead to abrupt switching in depth/blending order of Gaussians. This issue could potentially be addressed with anti-aliasing, which has been left as a topic for future research.

The algorithm proposed in this paper currently cannot incorporate any form of regularization. Introducing regularization could help better handle unseen areas and regions prone to artifacts.

Although the same hyperparameters are used across all evaluations, initial experiments suggest that reducing the position learning rate might be necessary for convergence in very large scenes (e.g., urban datasets).

Compared to prior point-based approaches, this method is highly compact, but it still uses significantly more memory than NeRF-based solutions. Training large scenes can cause peak GPU memory usage to exceed 20 GB. However, implementing low-level optimization logic could substantially reduce this. Rendering a trained scene requires sufficient GPU memory to store the entire model, with additional memory demands for the rasterizer ranging from 30 to 5000 MB depending on the scene size and image resolution.
