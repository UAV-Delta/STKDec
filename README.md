# STKDec
A pytorch implementation for the paper: Knowledge-enhanced Denoising Diffusion
for Cross-city Battery Swap Demand Prediction.

# Introduction
### Problem describetion
<img src="https://github.com/UAV-Delta/STKDec/blob/main/img/problem.pic.jpg" width="800"/>

We consider two cities, i.e. the source city and the target city. In the source city, the battery swap station network has already been deployed. In the target city, a battery swap station deployment plan is in place, and we aim to predict the corresponding battery swap demands for this plan before actual deployment. 

### Illustration of local diffusion model training on each client
<img src="https://github.com/UAV-Delta/STKDec/blob/main/img/framework.pic.jpg" width="400" />

The local model training on each client consists of two phases: (1) the forward diffusion phase taking the latent representations extracted by the missingness-tolerant masked autoencoder as input; and (2) the reverse denoising phase conditioned on a UKG-based urban environment characterization.

### Illustration of local diffusion model training on each client
<img src="https://github.com/UAV-Delta/STKDec/blob/main/img/method.pic.jpg" width="600" />

The local model training on each client consists of two phases: (1) the forward diffusion phase taking the latent representations extracted by the missingness-tolerant masked autoencoder as input; and (2) the reverse denoising phase conditioned on a UKG-based urban environment characterization.

# Installation
### Environment
1. Tested OS: Windows 11.
2. Python >= 3.9.
3. torch == 2.0.0.

### Dependencies
1. Install Pytorch with the correct CUDA version.
2. Use the pip install -r requirements.txt command to install all of the Python modules and packages used in this project.
