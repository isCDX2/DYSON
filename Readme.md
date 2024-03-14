# DYSON

The official PyTorch implementation of our **CVPR** 2024 paper:

**DYSON: Dynamic Feature Space Self-Organization for Online Task-Free Class Incremental Learning**

GitHub maintainer: [Yingjie Chen](https://github.com/isCDX2)

## Highlight
- --
### Brief Introduction
In this paper, we focus on a challenging Online Task-Free Class Incremental Learning (OTFCIL) problem. Different from the existing methods that continuously learn the feature space from data streams, we propose a novel compute-and-align paradigm for the OTFCIL. It first computes an optimal geometry, i.e., the class prototype distribution, for classifying existing classes and updates it when new classes emerge, and then trains a DNN model by aligning its feature space to the optimal geometry. To this end, we develop a novel Dynamic Neural Collapse (DNC) algorithm to compute and update the optimal geometry. The DNC expands the geometry when new classes emerge without loss of the geometry optimality and guarantees the drift distance of old class prototypes with an explicit upper bound. On this basis, we propose a novel Dynamic feature space Self-Organization (DYSON) method containing three major components, including 1) a feature extractor, 2) a Dynamic Feature-Geometry Alignment (DFGA) module aligning the feature space to the optimal geometry computed by DNC and 3) a training-free class-incremental classifier derived from the DNC geometry.  Experimental comparison results on four benchmark datasets, including CIFAR10, CIFAR100, CUB200, and CoRe50, demonstrate the efficiency and superiority of the DYSON method.

### Strong Performance
Our proposed DYSON steadily and dramatically outperforms the state-of-the-art methods.

|   Method      | backbone | buffer | cifar10,step=1 | cifar10,step=2 | cifar10,step=Gaussian | buffer | cifar100,step=1 | cifar100,step=5 | cifar100,step=Gaussian |
|:-------------:|:--------:|:------:|:--------------:|:--------------:|:---------------------:|:------:|:---------------:|:---------------:|:----------------------:|
|   ODDL        | ResNet18 | 2k     | -              | 52.7           | -                     | 2k     | -               | 27.2            | -                      |
|   SDP         | ResNet18 | 500    | -              | 66.2           | 76.3                  | 2k     | -               | -               | -                      |
|  DYSON(ours)  | ResNet18 | 0      | **73.4**       | **74.4**       | **76.5**              | 0      | **49.6**        | **45.3**        | **47.0**               |
|   CoPE        | ResNet50 | 1k     | -              | 48.9           | -                     | 5k     | -               | 21.6            | -                      |
|   CN-DPM      | ResNet50 | 1k     | -              | 45.2           | -                     | 1k     | -               | 20.1            | -                      |
|   GMED        | ResNet50 | 5k     | -              | 47.7           | -                     | 5k     | -               | 19.6            | -                      |
|   FCA         | ResNet50 | 0      | 77.5           | 78.7           | 76.1                  | 0      | 53.3            | 48.7            | 53.4                   |
|   Ensemble    | ResNet50 | 2k     | 78.3           | 79.0           | 50.1                  | 6k     | 54.1            | 55.3            | 39.0                   |
|   DSDM        | ResNet50 | 2k     | 79.4           | 79.6           | 78.7                  | 6k     | 54.9            | 55.3            | 55.5                   |
|**DYSON(ours)**| ResNet50 | 0      | **79.5**       | **80.7**       | **79.1**              | 0      | **58.9**        | **59.2**        | **58.6**               |
|  L2P          | ViT-S/8  | 1k     | 46.8           | 61.4           | 57.5                  | 3k     | 8.4             | 27.3            | 48.7                   |
|  DSDM         | ViT-S/8  | 1k     | 85.5           | 85.6           | 84.9                  | 3k     | 61.1            | 60.8            | 64.1                   |
|**DYSON(ours)**| ViT-S/8  | 0      | **92.6**       | **93.5**       | **93.8**              | 0      | **77.7**        | **75.6**        | **76.4**               |



### install all environment
Use the Anaconda (cu11)
```
conda env create -f DYSON_env.yaml
```
### Data Preparation
CIFAR10 and CIFAR100 data sets will be downloaded automatically, [CoRE50](http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip) and [CUB200](https://data.caltech.edu/records/65de6-vp158) data sets will need to be downloaded manually and put them in the ./DYSON/dataset
## Training and Evaluation
Only the operation of the default protocol is shown here, Adjust the specific parameters to change the protocol
- CIFAR10
```
python ./DYSON/main.py --data_name cifar10 --epoch 1
```
- CIFAR100
```
python ./DYSON/main.py --data_name cifar100 --epoch 1
```
- CoRe50
```
python ./DYSON/core.py
python ./DYSON/main.py --data_name core50 --epoch 1
```
- CUB200
```
python ./DYSON/main.py --data_name cub200 --epoch 20
```

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@InProceedings{He_2024_CVPR,
    author    = {He, Yuhang and Chen, YingJie and Jin, Yuhan and Dong, Songlin and Wei, Xing and Gong, Yihong},
    title     = {DYSON: Dynamic Feature Space Self-Organization for Online Task-Free Class Incremental Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {}
}
```

## Contact

If you have any questions or concerns, feel free to open issues or directly contact me through the ways on my GitHub homepage **provide below paper's title**.
