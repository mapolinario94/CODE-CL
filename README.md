# CODE-CL: Conceptor-Based Gradient Projection for Deep Continual Learning

This repository is the official implementation of CODE-CL, a conceptor-based gradient projection method for deep continual learning, and replicates the experimental results obtained on the SplitCIFAR100, SplitMiniIMAGENET, and  5-Dataset benchmarks. Its primary purpose is to aid in understanding the methodology and reproduce essential results. This work has been accepted at **the International Conference on Computer Vision, ICCV 2025**.

[[ICCV Paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Apolinario_CODE-CL_Conceptor-Based_Gradient_Projection_for_Deep_Continual_Learning_ICCV_2025_paper.html)

## Abstract
Continual learning (CL) — the ability to progressively acquire and integrate new concepts — is essential to intelligent systems to adapt to dynamic environments. However, deep neural networks struggle with catastrophic forgetting (CF) when learning tasks sequentially, as training for new tasks often overwrites previously learned knowledge. To address this, recent approaches constrain updates to orthogonal subspaces using gradient projection, effectively preserving important gradient directions for previous tasks. While effective in reducing forgetting, these approaches inadvertently hinder forward knowledge transfer (FWT), particularly when tasks are highly correlated. In this work, we propose Conceptor-based gradient projection for Deep Continual Learning (CODE-CL), a novel method that leverages conceptor matrix representations, a form of regularized reconstruction, to adaptively handle highly correlated tasks. CODE-CL mitigates CF by projecting gradients onto pseudo-orthogonal subspaces of previous task feature spaces while simultaneously promoting FWT. It achieves this by learning a linear combination of shared basis directions, allowing efficient balance between stability and plasticity and transfer of knowledge between overlapping input feature representations. Extensive experiments on continual learning benchmarks validate CODE-CL’s efficacy, demonstrating superior performance, reduced forgetting, and improved FWT as compared to state-of-the-art methods.

<p align = "center">
<img src = "method_diagram.png">
</p>
<p align = "center">
Overview of CODE-CL.
</p>

## How to Use

1. Install the required dependencies listed in `requirements.txt`. 
2. Use the following command to run an experiment:

    ```shell
    python main.py --param-name param_value
    ```

    A description of each parameter is provided in `./utils/args.py`.

## Reproducing Results in the Paper

To ensure reproducibility, we have provided a bash script (`./script.sh`) with all the commands used to obtain the results reported in the paper.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@InProceedings{Apolinario_2025_ICCV,
    author    = {Apolinario, Marco P. E. and Choudhary, Sakshi and Roy, Kaushik},
    title     = {CODE-CL: Conceptor-Based Gradient Projection for Deep Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {775-784}
}
```
