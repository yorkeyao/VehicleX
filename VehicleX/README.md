## VehicleX Engine

VehicleX contains 1,362 vehicles of various models in 3D with fully editable attributes. We also propose an attribute descent approach to let VehicleX approximate the attributes in real-world datasets.  

![fig1](https://github.com/yorkeyao/VehicleX/blob/master/VehicleX/Images/Platform.jpg)  

## News

* (1/2020) We have released the VehicleX source code.  
* (12/2019) We increased the vehicle numbers from 1,209 to 1,362 and labeled car types and color.  

## Requirements

* OS: Windows 10 or Ubuntu 14.04+.
* Python 3.6 only.
* Linux Server: X server with GLX module enabled.

## Running with VehicleX

The directly runnable version is available. We have both [windows version](https://drive.google.com/open?id=1cLKFhXc9HhKmsh05XrWSGKDs-GZ73_Hf) and [linux version](https://drive.google.com/open?id=1s7sZY17HCaPCENZI6SbxuNcBMWHgOkOU). You will also need [backgroud images](https://drive.google.com/file/d/1dx03ijDzJkbVp0XnZbvKLTYZSYMDJHsf/view?usp=sharing) prepared. Please download them and store in a file structure like this: 

```
~
└───VehicleX
    └───Build-win(is 'Build-linus' if you choose linux)
    │   │ VehicleX.exe(is 'VehicleX.x86_64' if you choose linux)
    │   │ ...
    │
    └───Background_imgs
    │   │ vdo (1).avi
    │   │ ...
    │
    └───inference.py
```

Besides, you will need to 

```python
pip install tensorflow-gpu==1.14
pip install mlagents==0.10.1
```
You can learning attributes from VehicleID dataset with attribute descent using

```python
python train.py --setting './VehicleID.json' --output './VehicleID-out.json'
```

That will save learned attribute to the output json file. After that, you can generate a dataset easily by running

```python
python inference.py --setting './VehicleID-out.json'
```

## Unity Development

If you wish to make changes to the Unity assets you will need to install the Unity Editor version. The source code for the engine itself has been released. Please see dir [./Unity_source](https://github.com/yorkeyao/VehicleX/tree/master/VehicleX/Unity_source).




