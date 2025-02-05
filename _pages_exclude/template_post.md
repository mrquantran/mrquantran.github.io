---
layout: post
title: Example
date: 2024-08-15
description: Example
tags: gaussian-splatting
categories: paper-notes
thumbnail: assets/img/paper-notes/gaussian_splatting/ejemplo_gaussian_splatting.png
disqus_comments: true
---

The paper begins by highlighting the recent revolution in novel-view synthesis using `Neural Radiance Field` methods, which renders 3D scenes from various viewpoints, given multiple images and their corresponding camera pose values

However, it points out two key problems: achieving high visual quality requires computationally expensive neural networks for training and rendering, and faster methods often sacrifice visual quality for speed. The problem is further compounded when rendering unbounded scenes at 1080p resolution, where no existing method can achieve real-time display rates.

This paper has become a hot topic in the `novel-view synthesis field`. It surpasses `Mip-NeRF360` (2022), the SOTA (state-of-the-art) method in rendering quality for high-resolution (1920x1080) outputs, and even reduces training time compared to `InstantNGP` (2022), the SOTA method for speed.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/paper-notes/gaussian_splatting/compared-nerf.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

> The reason this paper has gained significant attention is its revolutionary fast rendering speed (exceeding approximately 100 FPS), in addition to the advantages mentioned above.