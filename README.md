# My Installation Instructions

Get the repo for GeNN by cloning somewhere. Run `export PATH=export PATH=$PATH:$/Documents/genn-4.1.0/bin`. Then check if gcc and g++ is installed with `gcc --version` and `g++ --version`. Then navigate to `cd genn-4.1.0/` and run 
```python
make CPU_ONLY=1 DYNAMIC=1 LIBRARY_DIRECTORY=`pwd`/pygenn/genn_wrapper/
```
Then create a conda env, and install `numpy=1.14`. Then run `python setup.py develop`.
Also go to `/Documents/genn-4.1.0/pygenn/genn_model.py` and set `self.selected_gpu = selected_gpu` in the `__init__` method of GeNNModel to `self._selected_gpu = selected_gpu`.
