'use strict';

var assert = require('chai').assert,
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    ops = require('ndarray-ops'),
    blas2 = require('../index.js');

assert.ndCloseTo = function(a,b,tol) {
  assert(a.dimension===b.dimension, 'expected dimension '+a.dimension+' to equal dimension '+b.dimension+'.');
  assert.deepEqual(a.shape, b.shape, 'expected shape '+a.shape+' to equal shape '+b.shape+'.');
  var c = pool.zeros(a.shape, a.dtype);
  ops.sub(c,a,b);
  var err = ops.norm2(c);
  assert( err < tol, 'Expected error '+err+' to be less than tolerance '+tol+'.');
};

describe("BLAS Level 2",function() {

  var A, A0, x, x0, y, y0;

  beforeEach(function() {
    var Adata = [1,2,5,3];
    var xdata = [-4,7];
    var ydata = [3,-2];
    A = ndarray(new Float64Array(Adata), [2,2]);
    A0 = ndarray(new Float64Array(Adata), [2,2]);
    x = ndarray(new Float64Array(xdata));
    x0 = ndarray(new Float64Array(xdata));
    y = ndarray(new Float64Array(ydata));
    y0 = ndarray(new Float64Array(ydata));
  });

  it('gemv',function() {
    assert( blas2.gemv( -4, A, x, 2, y ) );
    assert.ndCloseTo( ndarray(new Float64Array([-34,-8])), y, 1e-8 );
    assert.ndCloseTo( A0, A, 1e-8 );
    assert.ndCloseTo( x0, x, 1e-8 );
  });

  it('trmv',function() {
    assert( blas2.trmv( A, x ) );
    assert.ndCloseTo( x, ndarray([10,21]), 1e-8 );
    assert.ndCloseTo( A0, A, 1e-8 );
  });

  it('trmv lower',function() {
    assert( blas2.trmv( A, x, true ) );
    assert.ndCloseTo( x, ndarray([-4,1]), 1e-8 );
    assert.ndCloseTo( A0, A, 1e-8 );
  });

  it('trsv',function() {
    assert( blas2.trsv( A, x ) );
    assert.ndCloseTo( x, ndarray([-8.66666666666666667,  2.3333333333333333]), 1e-8 );
    assert.ndCloseTo( A0, A, 1e-8 );
  });

  it('trsv lower',function() {
    assert( blas2.trsv( A, x, true ) );
    assert.ndCloseTo( x, ndarray([-4,  9]), 1e-8 );
    assert.ndCloseTo( A0, A, 1e-8 );
  });

});

