# ndarray-blas-level2

[![Build Status](https://travis-ci.org/scijs/ndarray-blas-level2.svg?branch=master)](https://travis-ci.org/scijs/ndarray-blas-level2) [![npm version](https://badge.fury.io/js/ndarray-blas-level2.svg)](http://badge.fury.io/js/ndarray-blas-level2)

BLAS Level 2 operations for [ndarrays](https://github.com/scijs/ndarray)


## Usage

This library implements the basic matrix-vector operations of the Level 2 Basic Linear Algebra Subprograms (BLAS).

Note: It's possible to accomplish the lower triangular functions with the upper triangular version plus flipping and unflipping dimensions, but that's a little convoluted. Instead, the lower triangular versions are suffixed with \_lower just to keep it really simple.

### `gemv( alpha, A, x, beta, y )`
Calculate `y <- alpha*A*x + beta*y`

### `trmv( A, x, isLower )`
Calculate `x <- A*x` for the upper triangular matrix A. Data below the diagonal is ignored. If `isLower` is true, uses the lower triangular portion of A instead.

### `trsv( A, x, isLower )`
Calculate `x <- A^-1 x` for the upper triangular matrix A. Data below the diagonal is ignored.  If `isLower` is true, uses the lower triangular portion of A instead.

## Credits
(c) 2015 Ricky Reusser. MIT License
