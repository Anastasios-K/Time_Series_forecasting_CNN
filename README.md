# Time_Series_forecasting_CNN
A time series forecasting model based on 1D CNN

## Anaconda environment set-up
Current conda version: 4.8.3
1) Create a conda environment, named "envname", with Python version 3.8.3
2) Activate the environment

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
CPU: Intel Core i5-9500 @ 3.00 Ghz, 6 cores  
GPU: Nvidia GeForce RTX 2070 Super  

The entire execution, including the grid search, approximatelly lasts 32 hours.  

The optimal model is designed after a grid search and a couple of refinement phases.  
Also, the learning progress proceeds from a well-designed annealing process.  

The project is to be completed shortly.  
So, some script files are to be added and the existing ones are to be updated.  
However, the existing files are fully functional and they can represent the main idea and a large part of the project.  

# How to run this:  
The main file calls whatever is required from the rest of the files.  
So, running only the main.py is enough.  

The Colab notebook just calls the main.py and runs it on Google Colab.  
If you prefer to run it on Colab, ignore the main.py and just use the Colab notebook.  
It is preferable for whoever does not have access to enough computational resources.
Although using a GPU on Google Colab is probably the best free way to run the whole project, the 12-hour time restriction, applied by Google, does not provide users with enough time to run the whole project at once.  
However, the main.py file is designed in that way, so everyone can easily break the entire process down into smaller partitions and concatenate the results at the end.
