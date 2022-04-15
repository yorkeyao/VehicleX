## VehicleX 

![fig1](https://github.com/yorkeyao/VehicleX/blob/master/VehicleX%20Interface/Images/VehicleX.jpg)  

This repository includes our code for the paper 'Simulating Content Consistent Vehicle Datasets with Attribute Descent' in ECCV2020. The VehicleX data is also with [CVPR AI City Challenge 2021](https://www.aicitychallenge.org/) and [Alice Challenge](http://45.32.72.229/).

Related material: [Paper](https://arxiv.org/abs/1912.08855), [Demo](http://vehiclex-demo.yaoy.cc:9610/), [6min Presentation](https://www.youtube.com/watch?v=JdA9Y_kPfJ4), [1min Intro](https://www.youtube.com/watch?v=YfCK4wngUac).

You may play with our [Demo](http://vehiclex-demo.yaoy.cc:9610/) for a quick view of our data. This demo contains 70 ids out of 1362. The whole procedure of using VehicleX images is in three steps:

1) Attribute distribution learning and generate images by VehicleX engine with learned attributes (content level domain adaptation); 

2) Perform style level domain adaptation (SPGAN) for generated images; 

3) Train a re-ID model for style transfered images. 

The code for the three steps is available on [./VehicleX Interface](https://github.com/yorkeyao/VehicleX/tree/master/VehicleX%20Interface), [./StyleDA](https://github.com/yorkeyao/VehicleX/tree/master/StyleDA) and [./Re-ID](https://github.com/yorkeyao/VehicleX/tree/master/Re-ID) respectively. You are welcomed to use every single part of the code for your research purpose. Unity source code for VehicleX is also available on [./Unity Source](https://github.com/yorkeyao/VehicleX/tree/master/Unity%20Source).

## VehicleX Adapted Images  

We make generated images from VehicleX directly. We have performed domain adaptation (both content level and style level) from VehicleX to VeRi-776, VehicleID and CityFlow respectively. They can be used to augment real-world data. The adaptated images can be downloaded the tabel below. 

|              | VeRi-776         | VehicleID        | CityFlow  |
|--------------|------------------|------------------|-----------|
| w Style      | [Baidu](https://pan.baidu.com/s/1q8t4mLGNVScjZevHFneVpw)(pwd:nz36),[Google](https://drive.google.com/file/d/1wLmUWY5clm88Jcmu1e5ITMYNCht_mnds/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1U16Z3GmTzEs-H_TrV24qlA)(pwd:akjh),[Google](https://drive.google.com/file/d/1C6VAf_Z19HuVPuUlb738HPRxpZKwWGx_/view?usp=sharing) | [Website](https://www.aicitychallenge.org/2020-track2-download/) |

The adapted image for CityFlow is used in AI City Challenge 2020 track 2. You may need to sign up to get CityFlow dataset access. The labeling format for all vehicleX data is "id_cam_num.jpg":

Taking "0001_c001_33.jpg" as an example: 
*  0001 means the vehicle id is 0001
*  c001 means the camera id is 001 
*  33 is the counting number

We also provide detailed labeling include vechile orientation, light intensity, light direction, camera distance, camera height, vehicle type and vehicle color in the XML file. The detailed labeling allows [multi task learning](https://github.com/NVlabs/PAMTRI).   

We released our joint training code (two stage training) with VeRi-776, VehicleID and CityFlow in [./Re-ID](https://github.com/yorkeyao/VehicleX/tree/master/Re-ID)

## VehicleX Engine (Unity-python Interface)

We provide a Unity-Python Interface, which you may generate your own images from python code without modifying Unity Environment or C# programming. You can perform attribute learning using attribute descent and then generate vehicle data with learned attributes. Please check [./VehicleX Interface](https://github.com/yorkeyao/VehicleX/tree/master/VehicleX%20Interface) for more details. 

## VehicleX Source Code and 3D Vehicle Models

If you want to make modification to the 3D environment or use our 3D vehicle models only (i.e. for 3D related project). We provide .fbx format vehicle models, which you may import to Unity, Unreal or Blender. We also released entire Unity project. Please check [./Unity Source](https://github.com/yorkeyao/VehicleX/tree/master/Unity%20Source) for more details.

If you find this code useful, please kindly cite:

```
@inproceedings{yao2020simulating,
  title={Simulating Content Consistent Vehicle Datasets with Attribute Descent},
  author={Yao, Yue and Zheng, Liang and Yang, Xiaodong and Naphade, Milind and Gedeon, Tom},
  booktitle={ECCV},
  year={2020}
}
@inproceedings{tang2019pamtri,
  title={Pamtri: Pose-aware multi-task learning for vehicle re-identification using highly randomized synthetic data},
  author={Tang, Zheng and Naphade, Milind and Birchfield, Stan and Tremblay, Jonathan and Hodge, William and Kumar, Ratnesh and Wang, Shuo and Yang, Xiaodong},
  booktitle={ICCV},
  year={2019}
}
```

If you have any question, feel free to contact yue.yao@anu.edu.au



