
# Latent Diffusion Model for 2D Chest CT Synthesis

This is a code repository for the [PHASE IV AI project](https://www.phase4ai-project.eu/) developed by [Eurecat - Multimedia Technologies Unit](https://multimedia-eurecat.github.io/team/).


---

We employed a Latent Diffusion Model (LDM) to generate 2D CT slices showing at least one pulmonary nodule.

Paper Title: [*Characterization of Synthetic Lung Nodules in Conditional Latent Diffusion of Chest CT Scans*](https://ebooks.iospress.nl/pdf/doi/10.3233/FAIA240408)

Presented at the 26th International Conference of the Catalan Association for Artificial Intelligence ([CCIA 2024](https://ccia2024.salleurl.edu/)).

> **Abstract:** *This study delves into the characterization of synthetic lung nodules using latent diffusion models applied to chest CT scans. Our experiments involve guiding the diffusion process by means of a binary mask for localization and various nodule attributes. In particular, the mask indicates the approximate position of the nodule in the shape of a bounding box, while the other scalar attributes are encoded in an embedding vector. The diffusion model operates in 2D, producing a single synthetic CT slice during inference. The architecture comprises a VQ-VAE encoder to convert between the image and latent spaces, and a U-Net responsible for the denoising process. Our primary objective is to assess the quality of synthesized images as a function of the conditional attributes. We discuss possible biases and whether the model adequately positions and characterizes synthetic nodules. Our findings on the capabilities and limitations of the proposed approach may be of interest for downstream tasks involving limited datasets with non-uniform observations, as it is often the case for medical imaging.*

Bibtex citation:
```
@incollection{mari2024characterization,
  title={Characterization of Synthetic Lung Nodules in Conditional Latent Diffusion of Chest CT Scans},
  author={Mar{\'\i} Molas, Roger and Sub{\'\i}as-Beltr{\'a}n, Paula and Pitarch Abaigar, Carla and Galofr{\'e} Cardo, Mar and Redondo Tejedor, Rafael},
  booktitle={Artificial Intelligence Research and Development},
  pages={44--51},
  year={2024},
  publisher={IOS Press}
}
```
DOI: https://doi.org/10.3233/FAIA240408

#### Installation
```
bash setup_ccia24_venv.sh
conda activate ccia24
```


---
#### Training

Use `CCIA24_run_exp.sh` to train the models presented in the paper.

Remember to update the .sh file with your own paths.

```
conda activate ccia24
bash CCIA24_run_exp.sh
```

---
#### Inference and evaluation

Use `generate_images.sh` to generate N images using one or more checkpoints.

Afterwards, you can evaluate the quality of synthetic images with respect to real images using `compute_fid.sh`.

Remember to update the .sh files with your own paths.


```
conda activate ccia24
bash generate_images.sh
bash compute_fid.sh
```

---
#### Other

The jupyter notebooks in `notebooks` were used for secondary debugging and data checks.

The scripts in `dataset_preprocessing` were used to produce the training set of 2D images and nodule masks in png format from the original LIDC-IDRI chest CT data.

---
#### Data

The training set of 2D images and nodule masks used in this work are available at Zenodo: https://doi.org/10.5281/zenodo.19053504

All data and model checkpoints (6.4 GB) can be downloaded using `download_data.sh`

```
bash download_data.sh
```

---

#### ThreeFold / TFGrid runtime

For deployment-specific docs (ThreeFold Dockerfiles, zinit runtime, environment variables, auth, and troubleshooting), see:

- [`docs/threefold-runtime.md`](docs/threefold-runtime.md)

---

#### Acknowledgements

Training images and nodule masks were derived from the original [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) dataset of chest CT scans (133.16GB).


Training script based on:
- Training an unconditional LDM: [https://github.com/zyinghua/uncond-image-generation-ldm](https://github.com/zyinghua/uncond-image-generation-ldm) - Original README [here](old_README.md).

UNet implementation based on:
- InverseSR: 3D Brain MRI Super-Resolution Using a Latent Diffusion Model: [https://github.com/BioMedAI-UCSC/InverseSR](https://github.com/BioMedAI-UCSC/InverseSR)
