# EAGLE: Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization

This repository is the official implementation of "[Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization (EAGLE)](https://openreview.net/forum?id=n8JWIzYPRz)" accepted by the 37th Conference on Neural Information Processing Systems (NeurIPS 2023).

[![Black Logo](framework.png)](https://openreview.net/forum?id=n8JWIzYPRz)
<p align="center">
  <a href="https://github.com/RingBDStack/EAGLE/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-orange" /></a> &nbsp; &nbsp;
  <a href="https://github.com/ambv/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a> &nbsp; &nbsp;
  <img src="https://img.shields.io/github/stars/RingBDStack/EAGLE?color=yellow&label=Star" alt="Stars" > &nbsp; &nbsp;
  <img src="https://img.shields.io/github/forks/RingBDStack/EAGLE?color=blue&label=Fork" alt="Forks" >
</p>

------

## 0. Abstract

Dynamic graph neural networks (DGNNs) are increasingly pervasive in exploiting spatio-temporal patterns on dynamic graphs. However, existing works fail to generalize under distribution shifts, which are common in real-world scenarios. As the generation of dynamic graphs is heavily influenced by latent environments, investigating their impacts on the out-of-distribution (OOD) generalization is critical. However, it remains unexplored with the following two challenges: 1) How to properly model and infer the complex environments on dynamic graphs with distribution shifts? 2) How to discover invariant patterns given inferred spatio-temporal environments? To solve these challenges, we propose a novel Environment-Aware dynamic Graph LEarning (EAGLE) framework for OOD generalization by modeling complex coupled environments and exploiting spatio-temporal invariant patterns. Specifically, we first design the environment-aware EA-DGNN to model environments by multi-channel environments disentangling. Then, we propose an environment instantiation mechanism for environment diversification with inferred distributions. Finally, we discriminate spatio-temporal invariant patterns for out-of-distribution prediction by the invariant pattern recognition mechanism and perform fine-grained causal interventions node-wisely with a mixture of instantiated environment samples. Experiments on real-world and synthetic dynamic graph datasets demonstrate the superiority of our method against state-of-the-art baselines under distribution shifts. To the best of our knowledge, we are the first to study OOD generalization on dynamic graphs from the environment learning perspective.

## 1. Requirements

Main package requirements:

- `CUDA == 10.1`
- `Python == 3.8.12`
- `PyTorch == 1.9.1`
- `PyTorch-Geometric == 2.0.1`

To install the complete requiring packages, use following command at the root directory of the repository:

```setup
pip install -r requirements.txt
```



## 2. Quick Start

### Training

To train the EAGLE, run the following command in the directory `./scripts`:

```train
python main.py --mode=train --use_cfg=1 --dataset=<dataset_name>
```
Explanations for the arguments:

- `use_cfg`: if training with the preset configurations.
- `dataset`: name of the datasets. `collab`, `yelp` and `act` are for Table 1, while `collab_04`, `collab_06`, and `collab_08` are for Table 2.


### Evaluation

To evaluate the EAGLE with trained models, run the following command in the directory `./scripts`:

```eval
python main.py --mode=eval --use_cfg=1 --dataset=<dataset_name>
```

Please move the trained model in the directory `./saved_model`. Note that, we have already provided all the pre-trained models in the directory for quick re-evaluation.

### Reproductivity

To reproduce the main results in Table 1 and Table 2, we have already provided all experiment logs in the directory ./logs/history. Run the following command in the directory ./scripts to reproduce the results in `results.txt`:

```
python show_result.py
```

## 3. Citation
If you find this repository helpful, please consider citing the following paper. We welcome any discussions with [yuanhn@buaa.edu.cn](mailto:yuanhn@buaa.edu.cn).

```bibtex
@inproceedings{yuan2023environmentaware,
  title={Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization},
  author={Yuan, Haonan and Sun, Qingyun and Fu, Xingcheng and Zhang, Ziwei and Ji, Cheng and Peng, Hao and Li, Jianxin},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=n8JWIzYPRz}
}
```

## 4. Acknowledgements

Part of this code is inspired by Zeyang Zhang et al.'s [DIDA](https://github.com/wondergo2017/DIDA). We owe sincere thanks to their valuable efforts and contributions.
