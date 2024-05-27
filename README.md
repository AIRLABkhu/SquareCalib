# Camera-LiDAR Extrinsic Calibration using Constrained Optimization with Circle Placement

## Abstract
```
Camera-LiDAR data fusion has demonstrated remarkable environmental perception capabilities in various fields.
The success of data fusion relies on the accurate matching of correspondence features from images and point clouds.
In this letter, we propose a target-based Camera-LiDAR extrinsic calibration by matching correspondences in both data.
Specifically, to extract accurate features from the point cloud, we propose a novel method that estimates the circle centers by optimizing the probability distribution from the initial position. 
This optimization involves generating the probability distribution of circle centers from circle edge points and using the Lagrangian multiplier method to estimate the optimal positions of the circle centers.
We conduct two types of experiments: simulations for quantitative results and real system evaluations for qualitative assessment. 
Our method demonstrates a $21\%$ improvement in simulation calibration performance for 20 target poses with LiDAR noise of $0.03\,m$ compared to existing methods, and also shows high visual quality in re-projecting point cloud onto images in real-world scenarios.
```
## Framework
![Framework](images/framework.png)

## Getting Started 

### 1.Requirements 
#### 1.1 Setup 
We was tested on Ubuntu 20.04 64-bit with ROS Noetic 
* Ubuntu 20.04 
* ROS Noetic
* Python 3.8
* Scipy 1.10.1

#### 1.2 Calibration Board  
We use a planar board with four circular hole placed on rectangular shaped, which has the same size of the ([velo2cam](https://github.com/beltransen/velo2cam_calibration))
<img src="images/calibration_target.png" alt="Framework" style="width:60%;">

### 2.LiDAR  

### 3.Camera
#### 3.1 Camera intrinsic parameters can be estimated using OpenCV ([zhang's method](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)) 
#### 3.2 Camera features is extracted using the ([lvt2calib](https://github.com/Clothooo/lvt2calib)) 

## Contact
For any questions, please contact to us at 
[kdh2769@khu.ac.kr](mailto:kdh2769@khu.ac.kr) or github **Issues** 
 
## Thanks

[1] J. Beltr ́an, C. Guindel, A. de la Escalera, and F. Garc ́ıa, “Automatic extrinsic calibration method for lidar and camera sensor setups,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 10, pp. 17 677–17 689, 2022

[2] J. Zhang, Y. Liu, M. Wen, Y. Yue, H. Zhang, and D. Wang, “L2v2t2 calib: Automatic and unified extrinsic calibration toolbox for different 3d lidar, visual camera and thermal camera,” in 2023 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2023, pp. 1–7

## Citation 
Please consider citing this work if you use our code in your research:
```
impending ...
```
```
SquareCalib
├─ README.md
├─ images
│  ├─ calibration_target.png
│  └─ framework.png
├─ sample_data
└─ src
   └─ velodnye_circle.py

```
```
SquareCalib
├─ README.md
├─ images
│  ├─ calibration_target.png
│  └─ framework.png
├─ sample_data
└─ src
   └─ velodnye_circle.py

```