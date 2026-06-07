# Variational Hard Concept Bottleneck (VHCB) layer

### Official implementation of the paper [A Probabilistic Hard Concept Bottleneck for Steerable Generative Models](https://openreview.net/pdf?id=Kcb6WufAco), accepted at ICLR 2026.

#### Abstract
Concept Bottleneck Generative Models (CBGMs) incorporate a human-interpretable concept bottleneck layer, which makes them interpretable and steerable. However, designing such a layer for generative models poses the same challenges as for concept bottleneck models in a supervised context, if not greater ones. Deterministic mappings from the model inner representations to soft concepts in existing CBGMs: (i) limit steerable generation to modifying concepts in existing inputs; and, more importantly, (ii) are susceptible to concept leakage, which hinders their steerability. To address these limitations, we first introduce the Variational Hard Concept Bottleneck (VHCB) layer. The VHCB maps probabilistic estimates of binary latent variables to hard concepts, which have been shown to mitigate leakage. Remarkably, its probabilistic formulation enables direct generation from a specified set of concepts. Second, we propose a systematic evaluation framework for assessing the steerability of CBGMs across various tasks (e.g., activating and deactivating concepts). Our framework which allows us to empirically demonstrate that the VHCB layer consistently improves steerability.

#### Acknowledgements
The code is based on the official implementations of [Concept Bottleneck Autoencoder (CB-AE)](https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm) and [Coded Discrete Variational Autoencoder (Coded DVAE)](https://github.com/mariamartinezgarcia/codedVAE). We gratefully acknowledge the authors for making their code publicly available.

---
## Environment

#### Conda environment installation

  ```
  conda create -n vhcb python=3.8
  conda install nvidia/label/cuda-11.7.0::cuda-nvcc cudatoolkit
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  pip install -r requirements.txt
  ```

#### CUDA runtime errors (StyleGAN)
  If you get CUDA runtime errors (during "Setting up PyTorch plugin...") when running the StyleGAN2, use this (from [CB-AE repo](https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm)):

  ```
  export CUDA_HOME=$CONDA_PREFIX
  export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
  export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
  ```

## Download base model checkpoints

Store the pretrained checkpoints for the base generative models in:

```text
models/checkpoints/pretrained_checkpoints
```


### Pretrained StyleGAN2 Models

#### CelebA-HQ

Source: [StyleGAN3 GitHub Repository](https://github.com/NVlabs/stylegan3?tab=readme-ov-file#additional-material)

```bash
# CelebA-HQ
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl
```

#### CUB

Source: [StyleGAN2-ADA PyTorch GitHub Repository](https://github.com/NVlabs/stylegan2-ada-pytorch)

```bash
# CUB
gdown https://drive.google.com/uc?id=1sW7WgvUFH2REZPQx88BjFneoItP9C0XB
```

## Download classifier checkpoints

Coming soon!

In the meantime, if you want to train or evaluate a model, you can either:

- Train your own classifiers (checkpoints should be stored in `models/clf_checkpoints/`), or  
- Use a CLIP-based zero-shot classifier.

## Training

Use `train_vhcb_stylegan.py` to train a VHCB model with a StyleGAN2 backbone.

### Key arguments

- **`-p | --pseudo-label`**  
  Specifies the model used to generate pseudo-labels (e.g., 'supervised' for ResNet18-based classifiers, with checkpoints stored in `models/clf-checkpoints/`, or 'clip' for zero-shot CLIP classifiers).

- **`-c | --config-file`**  
  Path to the configuration file (located in `configs/`) that defines the dataset, concepts, repetition codes, and training parameters.

- **`-cp_name | checkpoint-name`**  
  Name used to save the trained model checkpoint.

- **`--load-pretrained`**  
  Whether to initialize training from a pretrained VHCB checkpoint stored in `models/pretrained_checkpoints/` (useful for fine-tuning).

- **`--pretrained-load-name`**  
  Filename of the checkpoint to load from `models/pretrained_checkpoints/`.

### Configuration file

All other training settings—such as dataset selection, concepts to be used, side-channel dimensionality, repetition code configuration, number of epochs, batch size, and learning rate—are specified in the configuration file.


## Evaluation

### Unzip the latents used for evaluation!

#### CelebA-HQ
From the root directory:

```
mkdir -p eval_latents_rn50_celebahq

unzip eval_latents_rn50_celebahq_part1.zip -d eval_latents_rn50_celebahq
unzip eval_latents_rn50_celebahq_part2.zip -d eval_latents_rn50_celebahq
```
#### CUB
From the root directory:
```
unzip eval_latents_rn50_cub.zip
```

### Running evaluation

Use `evaluation_vhcb_stylegan.py` to evaluate a VHCB model with a StyleGAN2 backbone.

### Key arguments

- **`-c | --config-file`**  
  Path to the configuration file (located in `configs/`) that defines the dataset, concepts, repetition codes, and training parameters.


### Configuration file

All evaluation settings—such as dataset selection, concepts to use, side-channel dimensionality, repetition code configuration, classifiers used for evaluation (with checkpoints stored in models/clf-checkpoints/), and the model checkpoint to evaluate—are specified in the configuration file.

## Citation

Martínez-García, M., Alvarez, R. V., Lancho, A., Olmos, P. M., & Valera, I. (2026). A Probabilistic Hard Concept Bottleneck for Steerable Generative Models. In The Fourteenth International Conference on Learning Representations.

@inproceedings{martinez2026probabilistic,
  title={A Probabilistic Hard Concept Bottleneck for Steerable Generative Models},
  author={Mart{\'\i}nez-Garc{\'\i}a, Mar{\'\i}a and Alvarez, Ricardo Vazquez and Lancho, Alejandro and Olmos, Pablo M and Valera, Isabel},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}