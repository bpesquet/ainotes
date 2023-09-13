# AI notes

This repository contains the source and configuration files for my  notes on Artificial Intelligence.

## Dependencies and toolchain

This project is built with the following software:

- the [Python](https://www.python.org/) programming language ;
- [poetry](https://python-poetry.org/) for dependency management ;
- [Black](https://github.com/psf/black) for code formatting ;
- [pylint](https://github.com/pylint-dev/pylint) to detect programming mistakes before execution ;
- [PyTorch](https://pytorch.org/) for tensor management and Machine Learning algorithms.

Additionally, it uses the following tools to create the [website](https://www.bpesquet.fr/ainotes) from the source code:

- [jupytext](https://jupytext.readthedocs.io) to synchronize Python source files with [Jupyter](https://jupyter.org/) notebooks ;
- [Jupyter Book](https://jupyterbook.org) to generate a static website from the notebooks ;
- A [GitHub Action](.github/workflows/deploy.yaml) to publish the website to [GitHub Pages](https://pages.github.com/) after each code push.

When teaching, I run the notebooks locally and use [RISE](https://rise.readthedocs.io) to showcase them as live [reveal.js](https://revealjs.com)-based presentations.

## Usage

Once a new notebook is created in the `docs/` folder, it is automagically paired with a `.py` file by the same name in the `src/` folder. Afterwards, all updates to either file will be reflected into the other. This is configured in the [pyproject.toml](pyproject.toml) file.

Here are some useful commands for running this project:

```bash
# Check the code for mistakes
pylint src/

# Force resync of all notebook files (in docs/) with Python files (in src/)
# Useful after cloning the repo, since notebooks are excluded from version control
jupytext --sync src/*/*.py

# Build the website locally from notebooks files
# Alternative command: jb build docs/
# Output is in the docs/_build/ subfolder
jupyter-book build docs/
```
