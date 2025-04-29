## Set up environment
We recommend you to set up a conda environment for packages used in this homework.
```
conda create -n 2590-hw2 python=3.8
source activate 2590-hw2
pip install torch torchvision
pip install numpy
conda install -c anaconda ipykernel
```

## Jupyter notebook
If you want to use Jupyter notebook, remember to install `ipykernel`
and run `python -m ipykernel install --user --name=2590-hw2`
before opening the notebook.
In Jupyter, select `Kernel-->Change kernel-->2590-hw2`.

## Pytorch Tutorial
https://pytorch.org/tutorials/beginner/basics/intro.html