[![Python Versions](https://img.shields.io/pypi/pyversions/ainotes.svg)](https://pypi.org/project/ainotes)
[![PyPI Version](https://img.shields.io/pypi/v/ainotes.svg)](https://pypi.org/project/ainotes)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Artificial Intelligence Notes

A collection of interactive notes related to Artificial Intelligence. Built for educational purposes and updated (somewhat) regularly.

## Structure

This project has the following structure:

- [ainotes/](ainotes) contains the Python source files;
- [data/](data) contains the datasets used by some of the notes;
- [docs/](docs) contains the [Jupyter](https://jupyter.org/) notebooks synchronized with Python files and used to create the [online version](https://www.bpesquet.fr/ainotes) of the notes.

Once a new notebook (`.ipynb` file) is created in the `docs/` folder, it is automagically paired with a Python (`.py`) file by the same name in the `ainotes/` folder. Afterwards, all updates to either file will be reflected into the other. This is configured in the [pyproject.toml](pyproject.toml) file.

The notebooks are versioned in order to display their output online and open them in cloud execution platforms like [Google Colaboratory](https://colab.research.google.com/).

The project is also published as a [package](https://pypi.org/project/ainotes/) on [PyPI](https://pypi.org). This is necessary to import shared code in cloud execution platforms.

## Toolchain

This project is built with the following software:

- the [Python](https://www.python.org/) programming language;
- [Poetry](https://python-poetry.org/) for dependency management and deployment;
- [Black](https://github.com/psf/black) for code formatting;
- [Pylint](https://github.com/pylint-dev/pylint) to detect programming mistakes before execution;
- [Jupytext](https://jupytext.readthedocs.io) to synchronize Python source files with Jupyter notebooks;
- [Jupyter Book](https://jupyterbook.org) to generate a static website from the notebooks;
- [RISE](https://rise.readthedocs.io) to showcase notebooks as live [reveal.js](https://revealjs.com)-based presentations;
- A [GitHub Action](.github/workflows/deploy-book.yaml) to publish the website to [GitHub Pages](https://pages.github.com/);
- Another [GitHub Action](.github/workflows/publish-package.yaml) to publish the source code as a PyPI/TestPyPI package.

## Development notes

Here are some useful commands for running this project:

```bash
# Reformat all Python files
black ainotes/

# Check the code for mistakes
pylint ainotes/*

# Force resync of all notebook files (in docs/) with Python files (in ainotes/)
# Add the --execute flag to rerun all notebooks
jupytext --sync ainotes/**/*.py

# Build the website locally from notebook files
# Output is in the docs/_build/ subfolder
jupyter-book build docs/  # Or simply: jb build docs/

# Generate a PDF version of a chapter
# GIF files must be replaced by their static counterparts (PNG or JPG) in the notebook before launching this command
jupyter nbconvert --to PDF {notebook_file_name}
```

## License

Licenses are [Creative Commons](LICENSE) for the textual content and [MIT](CODE_LICENSE) for the code. See also [Acknowledgments](docs/reference/acknowledgments.md).

Copyright © 2023-present [Baptiste Pesquet](https://bpesquet.fr).
