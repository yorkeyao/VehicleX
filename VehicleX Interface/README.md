## VehicleX Engine

VehicleX contains 1,362 vehicles of various models in 3D with fully editable attributes. We also propose an attribute descent approach to let VehicleX approximate the attributes in real-world datasets.  

![fig1](https://github.com/yorkeyao/VehicleX/blob/master/VehicleX%20Interface/Images/Platform.jpg)  

## News

* (1/2021) We added support for change vehicle color. 
* (8/2020) We added attribute learning part and Posenet support. 
* (1/2020) We have released the VehicleX source code.  
* (12/2019) We increased the vehicle numbers from 1,209 to 1,362 and labeled car types and color.  

## Requirements

* OS: Windows 10 or Ubuntu 14.04+. 
* Python 3.6 only.
* Linux Server: X server with GLX module enabled.

## Running with VehicleX Unity-Python Interface

The directly runnable version is available. We have both [windows version](https://drive.google.com/file/d/1cLKFhXc9HhKmsh05XrWSGKDs-GZ73_Hf/view?usp=sharing) and [linux version](https://drive.google.com/file/d/1s7sZY17HCaPCENZI6SbxuNcBMWHgOkOU/view?usp=sharing). You will also need [backgroud images](https://drive.google.com/file/d/1dx03ijDzJkbVp0XnZbvKLTYZSYMDJHsf/view?usp=sharing) prepared. Please download them by click links above and store in a file structure like this: 

```
~
└───VehicleX
    └───Build-win(is 'Build-linus' if you use linux)
    │   │ VehicleX.exe(is 'VehicleX.x86_64' if you use linux)
    │   │ ...
    │
    └───Background_imgs
    │   │ vdo (1).avi
    │   │ ...
    │
    └───inference.py
```

For creating environment,

```python
conda create -n py36 python=3.6
conda activate py36
```

Besides, you will need to 

```python
pip install matplotlib
pip install torch torchvision
pip install tensorflow-gpu==1.14
pip install mlagents==0.10.1
pip install scikit-image
pip install scipy==1.0
```
For a quick environment check, you can learning attributes from VehicleID extracted features with attribute descent using

```python
python train.py --setting './settings/VehicleID.json' --output './settings/VehicleID-out.json'
```

That will save learned attribute to the output json file. After that, you can generate a dataset easily by running

```python
python inference.py --setting './settings/VehicleID-out.json'
```

This will generate a dataset using origrinal 1,362 ids. If you want to generate vehicles with random colors, please download this [package](https://drive.google.com/file/d/10zuKlpqWnd5uaPFcmOXq6pOuZmGpi5W3/view?usp=sharing) and run: 

```python
python random_color_inference.py --setting './settings/VehicleID-out.json'
```

## Training with Real Data

You will need real data prepared. We provide the configure json file for VehilceID and VeRi. You may need to download these datasets from their home pages ([VeRi](https://github.com/JDAI-CV/VeRidataset), [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html)). Please download them and store in a file structure like this:  

```
~
└───VehicleX
    └───datasets
    │   └─── VehicleID_V1.0
    │   │   │ VehicleID.npz
    │   │   │ image
    │   │   │ train_test_split
    │   │   │ ...
    |   └─── VeRi
    │   │   │ image_train
    │   │   │ ...
    │   │ preprocess-veri.py
    │   │ ...
    └───inference.py
    │   
    │ 
```

Preprocessing is also required. Please go to the datasets folder and run 

```python
python preprocess-VID.py
```
or for VeRi
```python
python preprocess-veri.py
```
After this, you may perform attribute descent by running:

```python
python train.py --setting './settings/VehicleID.json' --output './settings/VehicleID-out.json'
```
or for VeRi
```python
python train.py --setting './settings/VeRi.json' --output './settings/VeRi-out.json'
```

if you do not follow the file structure above. You may need to change target paths in the json file. 

## Reinforcement Learning 

We also provide a reinforcement based method following [LTS](https://arxiv.org/abs/1810.02513v2) framework. You may run with:  

```python
python train_LTS.py --setting './settings/VehicleID-LTS.json' --output './settings/VehicleID-out.json'
```

## Posenet Support

Posenet may also used to calculate fd score. It is based from [HRNet](https://github.com/NVlabs/PAMTRI/tree/master/PoseEstNet) in CVPR 2019. We provide a trained model from VeRi to calculate fd. You may switch to posenet by

```python
python train.py --setting './settings/VehicleID-real.json' --output './settings/VehicleID-out.json' --FD_model 'posenet'
```

## Running with Linux Server

You may need to run with your GPU server. 

On the client side, please use ssh -X username@ip to run graphics applications remotely.  

On the server side, you need to config /etc/ssh/sshd_config to have [X11Forwarding yes](https://unix.stackexchange.com/questions/12755/how-to-forward-x-over-ssh-to-run-graphics-applications-remotely). 

## Unity Development

If you wish to make changes to the Unity assets you will need to install the Unity Editor. The [source code](https://drive.google.com/file/d/17Jn5iov3e1rkWgOhID5c2RCnGWTxiuWA/view?usp=sharing) for the engine itself has been released. Please see more details in page [./Unity source](https://github.com/yorkeyao/VehicleX/tree/master/Unity%20Source). We show how to configure the source code step by step. 




