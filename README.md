# DREMVCL
# Dependencies
* python == 3.7.7    
* pytorch == 1.7.1    
* pytorch-lightning==1.0.8    
* scikit-learn    
* pandas 
* numpy
# Installation Guide
Clone this GitHub repo and set up a new conda environment.
#  create a new conda environment
* conda create -n dremvcl python=3.7.7
* conda activate dremvcl
#  install requried python dependencies
* pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
* pip install pytorch-lightning==1.0.8
* pip install scikit-learn
* pip install pandas
# Datasets
* Fdataset and Cdataset https://github.com/BioinformaticsCSU/BNNR
* LRSSL https://github.com/linwang1982/DRIMC
# Usage
```python  
cd DREMVCL
python main.py 
```
# Device
RTX2080Ti
