# Variational Wasserstein-Gradient-Flow Filter

Implements a filtering approach with a variational update based on Wasserstein gradient flows

## Installation
 
 Create a conda environment
    
    conda create -n NAME python=3.9
    
 Then head to the cloned repository and execute
 
    pip install -e .
    
 ## Examples
 
 A filtering example on a stochastic volatility model
 
    python examples/wasserstein_filter/markov_stochastic_volatility_wf_sqrt.py
    
 ## Cite
 ```
 @inproceedings{corenflos2023variational,
  title={Variational Gaussian Filtering via Wasserstein Gradient Flows}, 
  author={Corenflos, Adrien and Abdulsamad, Hany},
  booktitle={2023 31st European Signal Processing Conference (EUSIPCO)}, 
  year={2023},
}
```
