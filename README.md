## VehicleX 

This repository include our code for paper 'Simulating Content Consistent Vehicle Datasets with Attribute Descent'. If you want to reproduce our results, you will need to 

1) Generate images by VehicleX engine using attribute descent (content level damain apation); 

2) Perform style level damain apation; 

3) Train a Re-ID model. 

Code for these three steps is available on ./VehicleX, ./StyleDA and ./Re-ID saparatly. 

If you find this code useful, please kindly cite:

```
@article{yao2019simulating,
  title={Simulating Content Consistent Vehicle Datasets with Attribute Descent},
  author={Yao, Yue and Zheng, Liang and Yang, Xiaodong and Naphade, Milind and Gedeon, Tom},
  journal={arXiv preprint arXiv:1912.08855},
  year={2019}
}
@inproceedings{tang2019pamtri,
  title={Pamtri: Pose-aware multi-task learning for vehicle re-identification using highly randomized synthetic data},
  author={Tang, Zheng and Naphade, Milind and Birchfield, Stan and Tremblay, Jonathan and Hodge, William and Kumar, Ratnesh and Wang, Shuo and Yang, Xiaodong},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={211--220},
  year={2019}
}
```
