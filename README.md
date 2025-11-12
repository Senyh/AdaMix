# AdaMix

This repo is the PyTorch implementation of the paper:

**["Adaptive Mix for Semi-Supervised Medical Image Segmentation"](https://doi.org/10.1016/j.media.2025.103857)** 

## Usage

### 0. Requirements
The code is developed using Python 3.8 with PyTorch 1.11.0, and CUDA 11.3.
All experiments in our paper were conducted on a single NVIDIA A40 GPU with 48GB memory.

Install the main packages:
```angular2html
pytorch == 1.11.0
torchvision == 0.12.0
cudatoolkit == 11.3.1
```

### 1. Data Preparation
#### 1.1. Download data
The original data can be downloaded in following links:
* ACDC Dataset - [Link (Original)](https://www.creatis.insa-lyon.fr/Challenge/acdc/) - [Link (Processed)](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) 
* LA Dataset - [Link (Original)](http://atriaseg2018.cardiacatlas.org/) [Link (Processed)](https://github.com/yulequan/UA-MT/tree/master/data) 
* ISIC Dataset - [Link](https://challenge.isic-archive.com/)

PS: Please cite the papers of original datasets when using the data in your publications.


#### 1.2. Split Dataset
Following the list files (within the "data" folders) to split the datasets.

### 2. Training
```angular2html
python train_adamix_[st/mt/ct].py
```

### 3. Evaluation
```angular2html
python eval.py
```


## Citation
If you find this project useful, please consider citing:
```
@article{shen2025adaptive,
  title={Adaptive Mix for Semi-Supervised Medical Image Segmentation},
  author={Shen, Zhiqiang and Cao, Peng and Su, Junming and Yang, Jinzhu and Zaiane, Osmar R},
  journal = {Medical Image Analysis},
  pages = {103857},
  year = {2025},
  doi = {https://doi.org/10.1016/j.media.2025.103857}
}
```

## Contact
If you have any questions or suggestions, please feel free to contact me ([xxszqyy@gmail.com](xxszqyy@gmail.com)).