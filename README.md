# Orientation averages 

This is a repository for the python package orientation averages.

The package is by no means complete so please follow the development 
of the code in this repository. 

This package is the python version of the interpolator used in the publication: 

The folder statistics contains tests for runtime and accuracy.

## TODOS

- [x] runtime vs. accuracy analysis
- [ ] improve effiency -> don't resymmetrize if not necessary ?
- [x] write unit tests 
- [ ] figure out how to run a tester 
- [x] use frobenius norm as metric instead of geodesic?  

## Installation 

In the future this package will be available on PyPi. 
As of now it can be installed locally by executing:

```bash 
python setup.py bdist_wheel sdist
pip install .
```
