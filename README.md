<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# Open Medical Imaging Data Valuation

<div align="center">
  <a href="https://github.com/rsingla92/OpenMedImgDataVal">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">Open Medical Imaging Data Valuation</h3>
  <p align="center">
    Is a picture worth a thousand well-curated training samples?  
    <br />
    <a href="https://github.com/rsingla92/OpenMedImgDataVal/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/rsingla92/OpenMedImgDataVal/examples">View Examples</a>
    <a href="https://github.com/rsingla92/OpenMedImgDataVal/issues">Report Bug</a>
    <a href="https://github.com/rsingla92/OpenMedImgDataVal/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#datasets">Datasets</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project
Training data is the foundation of machine learning, yet not all data points are created equal. As models saturate in performance and as noisy or AI-generated content (“AI slop”) proliferates, the need for **principled methods to quantify the value of individual samples** is more pressing than ever. 

This repository explores and benchmarks **robust, scalable, and context-aware data valuation techniques** for machine learning pipelines, with an emphasis on **medical imaging datasets**.  

We build on and extend methods like **Shapley values**. Our goal is to provide the community with open implementations and evaluations that enable:  

* Efficient data curation for expensive annotation pipelines.  
* Identification of mislabeled, redundant, or harmful samples.  
* Task-aware data valuation for multi-task and medical ML models.  
* Exploration of group-wise effects (synergistic or antagonistic).  

Ultimately, this project is about enabling **better models with less data** without compromising rigor or reproducibility.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Technologies

Core stack and libraries we use include:

* [PyTorch](https://pytorch.org/) for deep learning
* [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) for numerical computation
* [scikit-learn](https://scikit-learn.org/) for baseline models and utilities
* [POT (Python Optimal Transport)](https://pythonot.github.io/) for OT methods
* [Giotto-TDA](https://giotto-ai.github.io/gtda-docs/) for topological data analysis

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

These steps assume your machine runs macOS and you have [Homebrew](https://brew.sh/) installed.

1. Clone project:

```sh
git clone https://github.com/ubcbmeg457/OpenMedImgDataVal.git
cd OpenMedImgDataVal
```

2. Install [uv](https://github.com/astral-sh/uv):

```sh
brew install uv
```

3. Set up project:

```sh
make setup
```

4. Test project setup:

```sh
uv run main.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage
Example: running Shapley value approximations on a toy dataset.
```sh
python examples/run_shapley.py --dataset mnist --approximation fastshap
```

Example: evaluating data valuation on a medical imaging dataset.
```
python examples/run_influence.py --dataset chestxray14 --model resnet18
```

See the examples folder for scripts and reproducible benchmarks. For more detailed usage, please refer to the documentation.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATASETS -->
## Datasets
We focus on publicly available datasets for reproducibility.
<a href="#">ChestX-ray14 (NIH)</a>: 100k+ frontal-view X-rays with 14 disease labels.
<a href="#">CheXpert (Stanford)</a>: Large dataset with multi-label uncertainty annotations.
<a href="#">MIMIC-CXR (PhysioNet)</a>: 370k chest radiographs with free-text reports.
<a href="#">MedMNIST</a>: Lightweight benchmark datasets for rapid prototyping.

Note: Due to licensing restrictions, datasets are not distributed in this repo. Please register and download them separately. Instructions for integration are in docs/datasets.md.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap
- [ ] ?
- [ ] ?
- [ ] ?
- [ ] ?
- [ ] ?
    - [ ] ?
    - [ ] ?

See the [open issues](https://github.com/rsingla92/OpenMedImgDataVal/issues) for a full list of proposed features (and known issues).
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REPOSITORY STRUCTURE -->
# Repository Structure
```
OpenMedImgDataVal/
│
├── data/                   # Placeholder for dataset links/download scripts
│   ├── README.md           # Instructions on how to fetch datasets
│
├── docs/                   # Documentation and tutorials
│   ├──                     # How to integrate external datasets
│   ├──                     # Explanation of implemented valuation techniques
│
├── examples/               # Example scripts for running experiments
│   ├── 
│   ├── 
│
├── openmedval/             # Core code
│   ├── __init__.py
│   ├── utils.py
│
├── tests/                  # Unit and integration tests
│
├── requirements.txt
├── LICENSE.txt
├── README.md
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
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

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact
Maintainer:
* Dr. Rohit Singla, MD PhD — [LinkedIn](https://www.linkedin.com/in/rsingla92/) - rsingla [at] ece [dot] ubc [dot] ca

Contributors:
* Dhairya Aggarwal - [@your_twitter](https://twitter.com/your_username)
* Chloe Christensen - [@your_twitter](https://twitter.com/your_username)
* Jaiden Siu - [GitHub](https://github.com/jaidensiu)
* Amy Yu - [@your_twitter](https://twitter.com/your_username)

Project Link: [https://github.com/rsingla92/OpenMedImgDataVal](https://github.com/rsingla92/OpenMedImgDataVal/)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* Prof. Tim Salcudean for infrastructure support
* The broader ML community for advancing research in data valuation
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/rsingla92/OpenMedImgDataVal.svg?style=for-the-badge
[contributors-url]: https://github.com/rsingla92/OpenMedImgDataVal/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/rsingla92/OpenMedImgDataVal.svg?style=for-the-badge
[forks-url]: https://github.com/rsingla92/OpenMedImgDataVal/network/members
[stars-shield]: https://img.shields.io/github/stars/rsingla92/OpenMedImgDataVal.svg?style=for-the-badge
[stars-url]: https://github.com/rsingla92/OpenMedImgDataVal/stargazers
[issues-shield]: https://img.shields.io/github/issues/rsingla92/OpenMedImgDataVal.svg?style=for-the-badge
[issues-url]: https://github.com/rsingla92/OpenMedImgDataVal/issues
[license-shield]: https://img.shields.io/github/license/rsingla92/OpenMedImgDataVal.svg?style=for-the-badge
[license-url]: https://github.com/rsingla92/OpenMedImgDataVal/blob/master/LICENSE.txt
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
