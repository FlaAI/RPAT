# Robust Perception Adversarial Training (RPAT)

- The paper has been accepted for publication at 2025 International Conference on Computer Vision (ICCV'25).

- Our code is divided into two individual parts, namely *RPAT_Benchmarks* and *RPAT_SOTAs*. The latter corresponds to the results in Sec 5.3 of our paper, while other results are acquired through the former.

- The implementation of *RPAT_Benchmarks* is based on https://github.com/alinlab/consistency-adversarial, while *RPAT_SOTAs* is built upon https://github.com/PKU-ML/ReBAT. Our great thanks for the generous open-source!

## Core Idea

<img decoding="async" src="Figs/roadmap.png" width="20%">


## 1. Dependencies

The two parts of code share the same environment, which can be reproduced via:

```
conda env create -f RPAT.yaml
```

## 2. For "RPAT_Benchmarks"

```
# Example for training with PGD-AT + RPAT
python train.py --mode adv_train --RA --model {model} --distance {norm} --epsilon {epsilon} --alpha {alpha} --epochs {epochs} --dataset {dataset}

# Example for training with Consistency-AT + RPAT
python train.py --mode adv_train --consistency --RA --model {model} --distance {norm} --epsilon {epsilon} --alpha {alpha} --epochs {epochs} --dataset {dataset}

# Example for evaluation with Auto-Attack
python eval.py --mode test_auto_attack --model {model} --distance {norm} --epsilon {epsilon} --dataset {dataset} --load_path {record_path}
```

## 3. For "RPAT_SOTAs"

```
# Example for training and evaluation with RPAT++ (i.e., ReBAT + RPAT)
python train_cifar_ra.py --fname {fname} --model {model} --norm {norm} --epsilon {epsilon} --pgd-alpha {alpha} --epochs {epochs} --num-classes {class_num_of_dataset}
```
