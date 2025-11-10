# STKDec
A pytorch implementation for the paper: Knowledge-enhanced Denoising Diffusion
for Cross-city Battery Swap Demand Prediction.

# Introduction
### Problem description
<img src="https://github.com/UAV-Delta/STKDec/blob/main/img/problem.pic.jpg" width="800"/>

We consider two cities, i.e. the source city and the target city. In the source city, the battery swap station network has already been deployed. In the target city, a battery swap station deployment plan is in place, and we aim to predict the corresponding battery swap demands for this plan before actual deployment. 

### Framework of STKDec
<img src="https://github.com/UAV-Delta/STKDec/blob/main/img/framework.pic.jpg" width="400" />

The framework of STKDec, a selective spatiotemporal knowledge-enhanced conditional diffusion model, which includes a forward diffusion module, a reverse denoising module, and a condition control module.

### Illustration of conditional control module in STKDec
<img src="https://github.com/UAV-Delta/STKDec/blob/main/img/method.pic.jpg" width="600" />

Design of the station context representation-based condition module(the upper panel), which comprises three components: (1) environment feature embedding, (2) inter-station interaction modeling, and (3) user behavior representation; The lower panel illustrates the implement of the cross-city demand prediction model within the target urban environment.

# Installation
### Environment
1. Tested OS: Windows 11.
2. Python >= 3.9.
3. torch == 2.0.0.

### Dependencies
1. Install Pytorch with the correct CUDA version.
2. Use the pip install -r requirements.txt command to install all of the Python modules and packages used in this project.
