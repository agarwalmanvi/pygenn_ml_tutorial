# My Installation Instructions for PyGeNN

Get the repo for GeNN by cloning somewhere. Run `export PATH=export PATH=$PATH:$/Documents/genn-4.1.0/bin`. 
Then check if gcc and g++ is installed with `gcc --version` and `g++ --version`. 
Then navigate to `cd genn-4.1.0/` and run 
```python
make CPU_ONLY=1 DYNAMIC=1 LIBRARY_DIRECTORY=`pwd`/pygenn/genn_wrapper/
```
Then create a conda env, and install `numpy=1.14`. Then run `python setup.py develop`. 
If `swig` is not installed, run
```python
sudo apt-get install python python-setuptools python-dev python-augeas gcc swig dialog
```
for some useful utilities in addition to swig. <br> <br>
You might have to go to `/Documents/genn-4.1.0/pygenn/genn_model.py` and set 
`self.selected_gpu = selected_gpu` in the `__init__` method of GeNNModel to `self._selected_gpu = selected_gpu`. 
\[**Update**: This was fixed in GeNN 4.2.0/PyGeNN 0.3.\] <br> <br>
This process should install most of the packages you need to get started. 
You may need to install `mlxtend` yourself using `pip install mlxtend`.
