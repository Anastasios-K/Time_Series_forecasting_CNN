# Time Series forecasting CNN
-- A time series forecasting model based on 1D CNN   
-- Stock price data is used  

## Important info
- Reproducible results are given
- Grid search - multiple phases
  - 1st -> 5 hyper-parameters
  - 2nd -> refinement stages
- Effect of different normalisation techniques
- Effect of different initialisation methods

## Anaconda environment set-up
Current conda version: 4.8.3   
**1)** Create a conda environment, named "envname", with Python version 3.8.3    
**2)** Activate the environment  

```bash
# 1)
conda create --name envname python=3.8.3
# 2)
conda activate envname
```
## Required packages installation
```bash
conda install tensorflow-gpu=2.1.0
conda install scicit-learn=0.23.2
conda install scipy=1.5.2
conda install pandas=1.1.3
conda install numpy=1.19.2
conda install matplotlib=3.3.2
conda install tqdm=4.50.2
```
## Computational resources and execution time
**CPU:** Intel Core i5-9500 @ 3.00 Ghz, 6 cores  
**GPU:** Nvidia GeForce RTX 2070 Super  

**Execution time, considering the grid search:** 32 hours approximatelly  

## Contribution
Pull requests are welcome. For major changes, please open an issue first

## Future development
- Further refinement and research  
- Ta-Lib python package integration
- Integration of Colab notebook to facilitate the execution (for those with limited computational recourses)   
