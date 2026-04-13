<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# Open Medical Imaging Data Valuation

Training data is the foundation of machine learning, yet not all data points are created equal. As models saturate in performance and as noisy or AI-generated content (“AI slop”) proliferates, the need for **principled methods to quantify the value of individual samples** is more pressing than ever. 

This repository explores and benchmarks **robust, scalable, and context-aware data valuation techniques** for machine learning pipelines, with an emphasis on **medical imaging datasets**.  

Our goal is to provide the community with open implementations and evaluations that enable:  

- Efficient data curation for expensive annotation pipelines
- Identification of mislabeled, redundant, or harmful samples
- Task-aware data valuation for multi-task and medical ML models
- Exploration of group-wise effects (synergistic or antagonistic)

Ultimately, this project is about enabling **better models with less data** without compromising rigor or reproducibility.

Beyond model performance, data valuation is a **sustainability lever**. Training on smaller, higher-quality subsets means fewer GPU hours, lower energy consumption, and reduced carbon emissions without sacrificing accuracy.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Technologies

Core stack and libraries we use include:

- [PyTorch](https://pytorch.org/) for deep learning (DenseNet121, 2D U-Net)
- [POT (Python Optimal Transport)](https://github.com/PythonOT/POT) for OT-based data valuation
- [KaggleHub](https://github.com/Kaggle/kagglehub) / [Synapse](https://www.synapse.org/) for dataset downloads

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials) (for dataset downloads)

### Local Setup

```sh
git clone https://github.com/ubcbmeg457/OpenMedImgDataVal.git
cd OpenMedImgDataVal
```

Install uv ([other methods](https://docs.astral.sh/uv/getting-started/installation/)):

```sh
# macOS / Linux
brew install uv
```

Install all dependencies:

```sh
make setup
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Running Pipelines

All pipelines are dispatched through a single entry point. The `--modality` and `--task` flags select the pipeline, and `--dv` selects the data valuation method.

```sh
# X-ray classification with KNN-Shapley
python src/main.py --modality xray --task class --dv shap

# X-ray classification with Optimal Transport
python src/main.py --modality xray --task class --dv ot

# MRI segmentation with KNN-Shapley
python src/main.py --modality mri --task seg --dv shap

# MRI segmentation with Optimal Transport
python src/main.py --modality mri --task seg --dv ot
```

For HPC clusters, see [jobs/README.md](jobs/README.md) for SLURM job scripts.

### Development

```sh
make format    # Auto-fix formatting and lint issues (ruff)
make lint      # Check formatting and lint without auto-fixing
make clean     # Remove venvs, caches, and build artifacts
make help      # Show all available targets
```

See [src/README.md](src/README.md) for pipeline architecture and [src/outputs/README.md](src/outputs/README.md) for output file documentation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Datasets

We focus on publicly available datasets for reproducibility.

> Note: Due to licensing restrictions, datasets are not distributed in this repo.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structure

```
OpenMedImgDataVal/
├── Makefile                    # Setup, formatting, and cleanup
├── pyproject.toml              # Root workspace config (uv + ruff)
├── uv.lock                     # Locked dependencies for reproducibility
├── .python-version             # Python version pin (3.11)
│
├── docs/                       # Documentation and roadmap
├── jobs/                       # SLURM batch scripts for HPC clusters
│
└── src/                        # Pipeline source code
    ├── main.py                 # Unified entry point (--modality, --task, --dv)
    ├── pyproject.toml          # Pipeline dependencies
    ├── xray_class/             # X-ray classification (NIH CXR-14 + DenseNet121)
    ├── mri_seg/                # MRI segmentation (BraTS 2023 + 2D U-Net)
    ├── dv/                     # Reusable data valuation methods
    │   ├── shap/knn_shapley.py # KNN-Shapley (Jia et al. 2019)
    │   └── ot/sinkhorn.py      # Sinkhorn OT (POT library)
    └── outputs/                # Generated outputs (per modality/task/method)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing
We welcome contributions from the community—whether it’s extending methods, adding datasets, improving documentation, or sharing benchmarks.
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/new-method`)
3. Commit your changes (`git commit -m 'Add new method'`)
4. Push to the branch (`git push origin feature/new-method`)
5. Open a Pull Request
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Maintainer:
- Dr. Rohit Singla, MD PhD — [LinkedIn](https://www.linkedin.com/in/rsingla92/) - rsingla [at] ece [dot] ubc [dot] ca

Contributors:
- Dhairya Aggarwal - [GitHub](https://github.com/DhairyaAggarwal02)
- Chloe Christensen - [GitHub](https://github.com/Chloechristensen)
- Jaiden Siu - [GitHub](https://github.com/jaidensiu)
- Amy Yu - [GitHub](https://github.com/amyyu799)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- Prof. Tim Salcudean for infrastructure support
- The broader ML community for advancing research in data valuation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ubcbmeg457/OpenMedImgDataVal.svg?style=for-the-badge
[contributors-url]: https://github.com/ubcbmeg457/OpenMedImgDataVal/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ubcbmeg457/OpenMedImgDataVal.svg?style=for-the-badge
[forks-url]: https://github.com/ubcbmeg457/OpenMedImgDataVal/network/members
[stars-shield]: https://img.shields.io/github/stars/ubcbmeg457/OpenMedImgDataVal.svg?style=for-the-badge
[stars-url]: https://github.com/ubcbmeg457/OpenMedImgDataVal/stargazers
[issues-shield]: https://img.shields.io/github/issues/ubcbmeg457/OpenMedImgDataVal.svg?style=for-the-badge
[issues-url]: https://github.com/ubcbmeg457/OpenMedImgDataVal/issues
[license-shield]: https://img.shields.io/github/license/ubcbmeg457/OpenMedImgDataVal.svg?style=for-the-badge
[license-url]: https://github.com/ubcbmeg457/OpenMedImgDataVal/blob/main/LICENSE.txt
