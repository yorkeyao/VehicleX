# SPGAN
This repo is based on Weijian's [SPGAN](https://github.com/Simon4Yan/eSPGAN), which is a great style level domain adaption technique. The code structure shows as follows:

```
~
└───StyleDA
    └───datasets
    │   └───sys2real
    |   │   │ trainA
    |   │   │ trainB
    |   │   │ testA
    |   │   │ testA
    └───code
    │   │ train_spgan.py
    │   │ test_spgan.py
    │   │ ...
```

You will need to get the dataset prepared first. Please fill the empty folder trainA, trainB, testA and testB in the dataset folder before running the code. Our trained model can be found at [google drive](https://drive.google.com/open?id=1bFX1KxNcBkyxWXdO_hOmP6t-GPATO9hK).
