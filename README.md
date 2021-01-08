# MAP Inference for Customized DPP via MIPS

MATLAB implementation for MAP Inference for Customized Determinantal Point Processes via Maximum Inner Product Search (AISTATS 2020, [paper](http://proceedings.mlr.press/v108/han20b/han20b.pdf), [video](https://slideslive.com/38930205/map-inference-for-customized-determinantal-point-processes-via-maximum-inner-product-search?ref=speaker-16952-latest)) 

  ## Installation
Run ```startup.m``` to add paths:
```matlab
>> install
```

## Usage
Run ```main.m``` for the synthetic experiments (See Section 4.1 in [paper](http://proceedings.mlr.press/v108/han20b/han20b.pdf))
```matlab
>> main
```

The hyperparameters (e.g., depth of trees, number of clusters and etc) can be set to other values. Check the line 4-17 in ```main.m```.