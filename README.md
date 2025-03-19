# RDVC: Efficient Deep Video Compression with Regulable Rate and Complexity Optimization

The official PyTorch implementation of our paper. If our open-source codes are helpful for your research, please cite our [paper](https://ieeexplore.ieee.org/document/10891391):

```
@ARTICLE{2025RDVC,
  author={Wei, Xiaojie and Lin, Jielian and Xu, Jiawei and Gao, Wei and Zhao, Tiesong},
  journal={IEEE Transactions on Multimedia}, 
  title={RDVC: Efficient Deep Video Compression with Regulable Rate and Complexity Optimization}, 
  year={2025},
  pages={1-12},
  doi={10.1109/TMM.2025.3543005}
}
```
## Test

- Dependency
  
  - see env.txt

- Please load the pre-training model weights ([Download link](https://ieeexplore.ieee.org/document/10891391)) first, and then run the test file Test_RDVC.py

## Train

- Download the training data. We train the models on the [Vimeo90k dataset](https://github.com/anchen1011/toflow) ([Download link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)).
- Run main.py to train the PSNR/MS-SSIM models. Configure the two-phase training parameters as required, referring to the training flow in Learner.py.
