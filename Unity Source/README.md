
## VehicleX Source Code 

VehicleX is based on [Unity](https://unity.com/). Here we released the whole [Unity project](https://drive.google.com/file/d/17Jn5iov3e1rkWgOhID5c2RCnGWTxiuWA/view?usp=sharing) include both **3D vehicle models** and **Unity python interface**. The project is stored in the google drive due to its size. [Background images](https://drive.google.com/file/d/1dx03ijDzJkbVp0XnZbvKLTYZSYMDJHsf/view?usp=sharing) are also required. Please download them by click links above. Once you download them the file structure should be like:

```
~
└───Source
    └───Assets
    │   │ Cars_folder
    │   │ ...
    │
    └───Background_imgs
    │   │ vdo (1).avi
    │   │ ...
    │
    └───Library
    │
    ...
```

[Unity hub](https://docs.unity3d.com/Manual/GettingStartedInstallingHub.html) is recommended to manage Unity projects. Please use Unity version 2019.3.0a8 or above. Once both the project and Unity editor are ready. The project can be opened easily by the Unity hub.    

![fig1](https://github.com/yorkeyao/VehicleX/blob/master/Unity%20Source/Images/unity_hub.PNG)  

## Image Generation by Unity Editor

* Open the project using Unity hub and you will see the interface below.
* Download the Unity python interface files.
* Comment out line 30 and uncomment line 34 for Inference.py in Unity python interface.  
* Run Inference.py and you will see a notice to press the play button. 
* Press the play button ▶️ in Unity Editor to get the images of Vehicles. 

![fig2](https://github.com/yorkeyao/VehicleX/blob/master/Unity%20Source/Images/interface.PNG) 

## Notice

* Please make sure the resolution of the game is 1920*1080. It can be controled in the game tab.
* If you see error CS1061: 'RawImage' does not contain a definition for 'm_Texture'. Please replace Unity/Hub/Editor/2019.3.0a8/Editor/Data/Resources/PackageManager/BuiltInPackages/com.unity.ugui/Runtime/UI/Core/Rawimages.cs with ./Script/RawImage.cs in this github project (Path can be different depends on your Unity version).
* Due to copyright issues. We are only able to release part of vehicle models in source code. The build version has all vehicle models included.  

## Separate 3D models

TBD


