# 3D-Seismic-Image-Fault-Segmentation
![demo](https://github.com/Jun-Tam/3D-Fault-Segmentation/raw/master/images/demo_application.gif)

### Summary
Train 3D CNN model with synthetic 3D seismic volumes containing faults.
Apply the trained model to actual seismic volume to detect faults.
Preliminary outcome. To be updated.

### Configuration
```
GPU: NVIDIA GeForce GTX 1080 Ti
Model architecture: Shallow 3D U-net
Training Data: 200
Validataion Data: 20
Batch size: 3 (Data Augmentation)
Data Augmentation: z-axis rotation (Randomly choose from 0, 90, 180, 270 deg.)
Feature size: 128 x 128 x 128
```

### Reference
FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation,Xinming Wu et al., Geophysics, 2019
