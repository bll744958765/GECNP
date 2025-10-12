# GECNP
This is a pytorch implementation of GECNP: A Gated-Expert Conditional Neural Process for Spatial Distribution Learning with Sparse Observations
## Experimental Setups
All experiments are implemented in the PyTorch framework and optimized using Adam. Training is performed on a single NVIDIA GeForce RTX
3080 GPU. 
##  Evaluation Metrics
To comprehensively evaluate the predictive performance of the proposed GECNP model, we adopt five commonly used regression metrics: mean absolute error (MAE), mean squared error (MSE), root mean
squared error (RMSE), coefficient of determination (R2), and negative log-likelihood (NLL). 
##  Experimental Study
This section presents experiments on synthetic datasets  spatial prediction tasks.
<img width="640" height="480" alt="Sampled Field" src="https://github.com/user-attachments/assets/9ea45214-d4ef-4039-a96e-5caef46acf7d" />
<img width="640" height="480" alt="True Field y" src="https://github.com/user-attachments/assets/b4a8628d-83c5-4843-9ecf-bd0ef2334b7c" />
## Run
You can run the train.py file to implement spatial prediction. In this file, you can also select different datasets and perform hyperparameter tuning.
