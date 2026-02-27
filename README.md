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

* Efficient data curation for expensive annotation pipelines.  
* Identification of mislabeled, redundant, or harmful samples.  
* Task-aware data valuation for multi-task and medical ML models.  
* Exploration of group-wise effects (synergistic or antagonistic).  

Ultimately, this project is about enabling **better models with less data** without compromising rigor or reproducibility.

Beyond model performance, data valuation is a **sustainability lever**. Training on smaller, higher-quality subsets means fewer GPU hours, lower energy consumption, and reduced carbon emissions without sacrificing accuracy.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Technologies

Core stack and libraries we use include:

- [CodeCarbon](https://github.com/mlco2/codecarbon) for tracking CO2 emissions
- [POT (Python Optimal Transport)](https://github.com/PythonOT/POT) for OT methods
- [Giotto-TDA](https://github.com/giotto-ai/giotto-tda) for topological data analysis

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

### JupyterHub / HPC Setup

On a cluster where JupyterHub is already running, you only need to install dependencies and register kernels:

```sh
make setup
make kernel
```

This registers a named Jupyter kernel for each subproject (e.g. "Python (xray-shapley-nb)"). Open JupyterHub and select the
kernel from the launcher or kernel picker.

To register a single module:

```sh
make kernel MODULE=xray-shapley-nb
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Running Notebooks

Start Jupyter Lab locally for a specific pipeline:

```sh
make notebook MODULE=xray-shapley-nb
make notebook MODULE=prototype
```

Or use the shortcuts:

```sh
make notebook-xray-shapley-nb
make notebook-prototype
```

On JupyterHub, open the notebook file directly and select the registered kernel.

### Development

```sh
make format    # Auto-fix formatting and lint issues (ruff)
make lint      # Check formatting and lint without auto-fixing
make clean     # Remove venvs, caches, and build artifacts
make help      # Show all available targets
```

See individual subproject READMEs for pipeline-specific documentation:
- [xray-shapley-nb/README.md](xray-shapley-nb/README.md) — SHAP-based data valuation for chest X-ray classification (notebook)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Datasets

We focus on publicly available datasets for reproducibility.

> Note: Due to licensing restrictions, datasets are not distributed in this repo.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structure

```
OpenMedImgDataVal/
├── Makefile                # Setup, notebooks, formatting, and cleanup
├── pyproject.toml          # Root workspace config (uv + ruff)
├── uv.lock                 # Locked dependencies for reproducibility
├── .python-version         # Python version pin (3.11)
│
├── xray-shapley-nb/        # SHAP-based data valuation for chest X-rays (notebook)
│   ├── pyproject.toml      # Subproject dependencies
│   ├── xray_shapley.ipynb  # End-to-end pipeline notebook
│   └── README.md           # Pipeline walkthrough and results
│
├── prototype/              # Prototyping and experimentation
│   ├── pyproject.toml
│   └── prototype.ipynb
│
├── mri-seg/                # MRI brain segmentation
│   └── BRAINSEG.ipynb
│
├── docs/                   # Documentation and roadmap
└── src/                    # Shared library code (planned)
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
* Dr. Rohit Singla, MD PhD — [LinkedIn](https://www.linkedin.com/in/rsingla92/) - rsingla [at] ece [dot] ubc [dot] ca

Contributors:
* Dhairya Aggarwal - [GitHub](https://github.com/DhairyaAggarwal02)
* Chloe Christensen - [GitHub](https://github.com/Chloechristensen)
* Jaiden Siu - [GitHub](https://github.com/jaidensiu)
* Amy Yu - [GitHub](https://github.com/amyyu799)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

* Prof. Tim Salcudean for infrastructure support
* The broader ML community for advancing research in data valuation

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
