
# Description
This repository will contain the code of RTCnet from our manuscript entitled
***"CNN-based Road User Detection using the 3D Radar Cube"*** after it is accepted and published. The paper was submitted to Robotics and Automation Letters (RA-L).

# What is RTCnet?
RTCnet is a radar based, single-frame, multi-class detection method for moving road users (pedestrian, cyclist, car), which utilizes low-level radar cube data.
The method {provides} class information both on the {radar target- and object-level}. {Radar targets are classified} individually after extending the target features with a cropped block of the 3D radar cube around their positions, 
thereby capturing the motion of moving parts in the local velocity distribution. A Convolutional Neural Network (CNN) is proposed for this classification step. Afterwards, object proposals are {generated} with a class-{specific} clustering step, which not only considers the radar targets' positions and velocities, but their calculated class scores as well.

# Citing information
We will add citing information after publication.


# Network Structure
TO BE FILLED by the figure

# Dataset
The feature array for training is a numpy array, with n rows and m columns . Each row corresponds to a sample cropped from the entire radar cube by using radar target as ROI. Each column corresponds to a feature. If the cropped window size is 5 x 5 x 32 (range-angle-speed bins), there will be 800 features from the radar cube. In addition to the radar cube, the range, angle, radar cross section (RCS) and radial speed are appended, thus each sample has 804 features. 

The labels are saved in another numpy array, with n rows and 1 column. 0 means unknown. 1 means pedestrian. 2 means cyclist. 3 means cars. 

Due to NDA issue, the dataset is not disclosed. However, a pseudo dataset for running the network can be downloaded [HERE](). 

# Preparation

Clone the code into local machine
```bash
git clone https://github.com/tudelft-iv/RTCnet.git
```
Run bash file for the preparation
```bash
cd RTCnet/bash
bash setup.bash
```

# Train, Test, Instance Segmentation
```bash
cd RTCnet
python3 train_RTC_ensemble.py
python3 test_RTC_ensemble.py
cd ..
python3 instance_seg.py
```
