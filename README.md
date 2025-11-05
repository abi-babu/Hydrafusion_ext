# HydraFusion Extensions for ZOD and BS-Breath Datasets

This repository provides extensions and utilities to integrate the [HydraFusion model](https://github.com/AICPS/hydrafusion) with the [ZOD (Zenseact Open Dataset)](https://zod.zenseact.com/) and BS-Breath datasets. It includes preprocessing scripts, evaluation tools, and dataset-specific adaptations to support object detection and accuracy estimation tasks.

## Repository Contents

- `zod_process.py`  
  Preprocesses the ZOD Mini Sequence dataset by converting its input format into `.pkl` files. These files are used as input for the HydraFusion model. This script does not perform augmentation or filtering—only format conversion.

- `zod_eval.py`  
  Evaluates the HydraFusion model on the ZOD dataset. It performs object detection and computes accuracy metrics to assess model performance.

- BS-Breath dataset integration  
  Additional utilities and extensions for working with the BS-Breath dataset are included in this repository.

## ZOD Dataset Instructions

To preprocess the ZOD dataset:

1. Download the ZOD Mini Sequence dataset from the [Zenseact website](https://zod.zenseact.com/).
2. Run `zod_process.py` to convert the dataset into `.pkl` format compatible with HydraFusion.
3. Use `zod_eval.py` to evaluate the model's performance on the processed ZOD data.

## References

- HydraFusion model: [github.com/AICPS/hydrafusion](https://github.com/AICPS/hydrafusion)  
- ZOD dataset: [zod.zenseact.com](https://zod.zenseact.com/)

