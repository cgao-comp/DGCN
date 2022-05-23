#DGCN-TCYB2022
The code related to the paper below： Gao Chao and Zhu Junyou and Zhang Fan and Wang Zhen and Li Xuelong, A Novel Representation Learning for Dynamic Graphs Based on Graph Convolutional Networks


## Run

`train_dgcn.py` is used to execute a full training run.
After cd the file, using 'python train_dgcn.py' to run.

experiment for link prediction: please make sure isone_hot = True, islabel = False
experiment for node clustering: please make sure isone_hot = False, islabel = True

## Reference
If you make advantage of DGCN in your research, please cite the following in your manuscript:

```
@article{gao2022novel,
  title={A Novel Representation Learning for Dynamic Graphs Based on Graph Convolutional Networks},
  author={Gao, Chao and Zhu, Junyou and Zhang, Fan and Wang, Zhen and Li, Xuelong},
  journal={IEEE Transactions on Cybernetics},
  year={2022},
  doi={10.1109/TCYB.2022.3159661} 
  publisher={IEEE}
}
```