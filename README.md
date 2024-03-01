# Auxiliary Fourier Augmentation
This repository contains the code for the paper "Fourier-basis Functions to Bridge Augmentation Gap: Rethinking Frequency Augmentation in Image Classification" accepted at CVPR 2024.

## Introduction
We propose Auxiliary Fourier-basis Augmentation (AFA), a complementary technique targeting augmentation in the frequency domain and filling the robustness gap left by visual augmentations. 
We demonstrate the utility of augmentation via Fourier-basis additive noise in a straightforward and efficient adversarial setting.
Our results show that AFA benefits the robustness of models against common corruptions, OOD generalization, and consistency of performance of models against increasing perturbations, with negligible deficit to the standard performance of models over various benchmarks and resolutions. 
It can be seamlessly integrated with other augmentation techniques to further boost performance. 

For more details see our [CVPR 2024 paper: Fourier-basis Functions to Bridge Augmentation Gap: Rethinking Frequency Augmentation in Image Classification]()

## Schema

<img align="center" src="assets/schema.jpg" width="750">

## Contents

This directory includes a reference implementation in PyTorch and Numpy of the augmentation method used in AFA.

We also include PyTorch re-implementations of AFA on both CIFAR-10/100 and ImageNet which both support training and evaluation on CIFAR-10/100-C and ImageNet-C.
