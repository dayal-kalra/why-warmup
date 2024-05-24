# Repository Overview

This repository contains various Jupyter notebooks corresponding to various results of the main paper::

The library requirements can be installed using: pip3 install -q -r requirements.txt

## Notebooks

1. **four_regimes_fcn.ipynb**:
   - Description: Using FCNs as an example, the notebook shows that neural networks typically shows four training regimes, and studies the effect of initialization and parameterization.

2. **four_regimes_uv_model.ipynb**: 
   - Description: Demonstrates that the UV model trained on a single example (x, y) shows all four training regimes and captures the effect of initialization and parameterization.

3. **bifurcation_uv_model.ipynb**: 
   - Description: Shows that the UV model exhibits a period-doubling route to chaos as the learning rate is increased. The late time sharpness versus learning rate is referred to as the bifurcation diagram.

4. **weight_norm.ipynb**: 
   - Description: Shows that weight norm and sharpness are correlated during early training.

## Folders

- **utils**: 
  - Description: Contains utility scripts or modules used across various notebooks. Includes custom functions, data preprocessing tools, or visualization utilities.

- **data**: 
  - Description: A directory containing data gnerated by the notebooks.

## Other Files

- **requirements.txt**: 
  - Description: Lists all Python dependencies required to run the notebooks in this repository. Useful for setting up a consistent environment.

