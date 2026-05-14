# GECNP: A Gated Expert Conditional Neural Process for Spatial Distribution Learning with Sparse Observations

Official PyTorch implementation of GECNP (Gated Expert Conditional Neural Process) for sparse spatial prediction and uncertainty-aware spatial distribution learning.

GECNP integrates:
Conditional Neural Processes (CNPs) for context-to-target probabilistic prediction,

Mixture-of-Experts (MoE) for heterogeneous spatial representation learning,

Mixture Density Networks (MDNs) for multimodal predictive uncertainty modeling.

The model is specifically designed for spatial prediction tasks with:

sparse observations,

irregular spatial layouts,

heterogeneous spatial structures,

multimodal target distributions,

and uncertainty-aware inference requirements.
# Overview

Spatial prediction is a fundamental problem in geostatistics, environmental science, remote sensing, urban computing, and geoscience exploration. Traditional approaches such as Kriging and GWR provide strong statistical interpretability but often struggle with nonlinear relationships and heterogeneous spatial patterns. Deep learning methods improve representation learning capability but frequently lack principled uncertainty quantification.

## GECNP addresses these limitations by combining:

Spatially-aware gated expert encoding

Density-based probabilistic decoding

Context-target neural process learning

Multimodal predictive distribution modeling

The framework explicitly models:

spatial autocorrelation,

geographic attribute similarity,

multimodal spatial responses,

and predictive uncertainty.

# Framework Architecture

The proposed framework contains two major components:

## 1. Gated Expert Encoder (GEEnc)

The encoder learns target-specific latent spatial representations using:

Shared Expert (SE)：Captures global contextual structure shared across the entire spatial domain.

Distance Expert (DE)：Models spatial autocorrelation by emphasizing nearby spatial observations.

Similarity Expert (SimE)：Captures geographically nonlocal but structurally similar patterns using auxiliary attributes.

Gated Expert Fusion (GEF)：Dynamically balances distance-based and similarity-based representations through a learnable gating mechanism.

## 2. Density-Gated Mixture Decoder (DGMDec)

The decoder models the predictive distribution as a mixture of Gaussian components.

It consists of:

Density Experts:Multiple MLP-based Gaussian experts predicting: mean，variance for different latent spatial regimes.

Density Gating Network:Learns adaptive mixture weights for each target query.

This enables:
multimodal prediction,
heteroscedastic uncertainty modeling,
and calibrated probabilistic inference.

### Key Features
Spatially-aware neural process framework

Explicit modeling of spatial autocorrelation

Geographic similarity learning

Sparse observation adaptation

Mixture-of-Gaussians predictive distributions

Uncertainty quantification

Flexible context-target learning

End-to-end differentiable training

## Experimental Setups
All experiments are implemented using the PyTorch framework.

### Hardware
GPU: NVIDIA GeForce RTX 3080
CUDA-enabled training

### Software
Python 3.9+
PyTorch 2.x
NumPy
Scikit-learn
Matplotlib
Pandas 

##  Evaluation Metrics
To comprehensively evaluate the predictive performance of the proposed GECNP model, we adopt five commonly used regression metrics: mean absolute error (MAE), mean squared error (MSE), root mean
squared error (RMSE), coefficient of determination (R2), and negative log-likelihood (NLL). 
## Data
This method uses the following  datasets for method evaluation: 
Synthetic Dataset: We construct a regular 50×50 grid to generate spatial coordinates uniformly distributed in the domain [0, 1]×[0, 1]. This synthetic benchmark designed to evaluate: spatial generalization, multimodal prediction, and uncertainty calibration.

Election: Voting data related to the U.S. presidential election.This data was used to evaluate: heterogeneous spatial patterns, socioeconomic spatial effects, and nonstationary geographic relationships.

California Housing: The classic California housing dataset, used for regression tasks.

Chengdu_data: A real-world second-hand housing price dataset collected from Chengdu. This dataset evaluates: urban spatial heterogeneity, sparse regional sampling, real-estate price prediction.
##  Experimental Study
This section presents experiments on synthetic datasets  spatial prediction tasks.
<img width="640" height="480" alt="Sampled Field" src="https://github.com/user-attachments/assets/9ea45214-d4ef-4039-a96e-5caef46acf7d" />
<img width="640" height="480" alt="True Field y" src="https://github.com/user-attachments/assets/b4a8628d-83c5-4843-9ecf-bd0ef2334b7c" />
## Run
To run ablation experiments, you do not need to modify the core code structure. Please implement it in the main file-train.py of the corresponding method (especially GECNP) by modifying the startup command line parameters or directly editing the args configuration (argparse.ArgumentParser()) in the code
Example Training Configuration:
  python train.py \
    --dataset california \
    --epochs 500 \
    --batch_size 32 \
    --hidden_dim 128 \
    --num_experts 3 \
    --num_gaussians 5 \
    --lr 1e-3
    
  Ablation experiments can be performed directly by modifying:
  
    command-line arguments,
    or the argparse.ArgumentParser() configuration in train.py.
    
  Typical ablation settings include:
  
    Ablation	                Description
    
    Remove DE	                Remove distance expert
    
    Remove SimE	               Remove similarity expert
    
    Remove Gating              Replace adaptive fusion with fixed averaging
    
    Single Gaussian	           Replace MDN with single Gaussian decoder
    
    Remove Shared Expert	    Remove global contextual aggregation
    
## Figure
All charts used in this paper to display experimental results and model analysis are generated by the figure.ipynb file.
You can start Jupyter Notebook with the following command to view or regenerate these charts

