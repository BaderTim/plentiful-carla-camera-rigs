# Benchmark: Plentiful Carla Camera Rigs
[![Project Page](https://img.shields.io/badge/Project%20Page-green?logo=github&label=github.io)](https://badertim.github.io/plentiful-carla-camera-rigs/)
[![arXiv](https://img.shields.io/badge/1234.12345-red?logo=arxiv&logoColor=red&label=arXiv)](https://arxiv.org)
[![huggingface](https://img.shields.io/badge/Dataset-yellow?logo=huggingface&label=HuggingFace)](https://huggingface.co/datasets/timb2001/plentiful-carla-camera-rigs/)

This repository contains the research code of our paper `Understanding Cross-Rig Generalization in Automotive Perception: a Multi-Rig Benchmark and Rig Variation Metrics` (Accepted at ECCV 2026).

**Abstract**
> Camera-based perception systems for autonomous driving are typically developed and evaluated using fixed sensor rigs, while real-world vehicle fleets exhibit substantial variation in camera placement, orientation, field of view, and camera count. 
> This mismatch introduces a cross-rig domain gap in which only the geometric observation process changes.
>
> To study this effect under controlled conditions, we introduce Plentiful Carla Camera Rigs (PCCR), a 3D object detection benchmark that renders identical driving scenes under 14 systematically designed camera rigs. 
> Using this benchmark, we analyze cross-rig transfer behavior of representative multi-view perception architectures and observe substantial performance shifts induced by geometric rig variation. 



![rigs](./rigs.png) 
*The 14 Rigs of our Plentiful Carla Camera Rigs (PCCR) Dataset.*

The full generated dataset is available on Hugging Face: [timb2001/plentiful-carla-camera-rigs](https://huggingface.co/datasets/timb2001/plentiful-carla-camera-rigs/).

## Repository Structure

- [pccr/README.md](pccr/README.md) - dataset generation pipeline (CARLA scene generation, trajectory recording, scene replay/capture, conversion tools).
- [models/README.md](models/README.md) - baseline model setup and cross-rig train/test scripts for BEVDet, BEVFusion, Fast-BEV, and PETR.
- [metrics/README.md](metrics/README.md) - RigV/RigCD analysis scripts and paper reproduction pipeline from standardized model results.

## Citation
Please use the following BibTeX entry to cite our work: 
```
@inproceedings{bader2026crossrig,
    title     = {Understanding Cross-Rig Generalization in Automotive Perception: a Multi-Rig Benchmark and Rig Variation Metrics},
    author    = {Bader, Tim Alexander and Eberhardt, Tim Dieter and Dillitzer, Maximilian and Stork, Wilhelm},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2026},
    note      = {Project page: https://badertim.github.io/plentiful-carla-camera-rigs/}
}
```