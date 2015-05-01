# ndarray-blas-level2

[![Build Status](https://travis-ci.org/scijs/ndarray-blas-level2.svg?branch=master)](https://travis-ci.org/scijs/ndarray-blas-level2) [![npm version](https://badge.fury.io/js/ndarray-blas-level2.svg)](http://badge.fury.io/js/ndarray-blas-level2)

BLAS Level 2 operations for [ndarrays](https://github.com/scijs/ndarray)


## Usage

This library implements the basic vector operations of the Level 2 Basic Linear Algebra Subprograms (BLAS).


### `gemv( alpha, A, x, beta, y )`
Calculate `y <- alpha*A*x + beta*y`

### `trmv( A, x )`
Calculate `x <- A*x` for the upper triangular matrix A. Data below the diagonal is ignored. For a lower triangular 

### `trsv( A, x )`
Calculate `x <- A^-1 x` for the upper triangular matrix A. Data below the diagonal is ignored.

### `trsv_lower( A, x )`
Calculate `x <- A^-1 x` for the lower triangular matrix A. Data above the diagonal is ignored. Note: It's possible to accomplish this with the trsv function by flipping and unflipping dimensions, but that's just a little convoluted. Instead, this function simply performs forward-substitution instead of back-substitution.



## Credits
(c) 2015 Ricky Reusser. MIT License
