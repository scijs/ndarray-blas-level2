'use strict';

var chai = require('chai');
var assert = chai.assert;
var ndarray = require('ndarray');
var pool = require('ndarray-scratch');
var ops = require('ndarray-ops');
var blas2 = require('../index.js');

assert.ndCloseTo = function (a, b, tol) {
  assert(a.dimension === b.dimension, 'expected dimension ' + a.dimension + ' to equal dimension ' + b.dimension + '.');
  assert.deepEqual(a.shape, b.shape, 'expected shape ' + a.shape + ' to equal shape ' + b.shape + '.');
  var c = pool.zeros(a.shape, a.dtype);
  ops.sub(c, a, b);
  var err = ops.norm2(c);
  assert(err < tol, 'Expected error ' + err + ' to be less than tolerance ' + tol + '.');
};

var createSymmetricMatrix = function (n) {
  var elementMax = 100;
  var n2 = n * n;
  var A = ndarray(new Float64Array(n2), [n, n]);
  for (var i = 0; i < n; ++i) {
    for (var j = 0; j < n; ++j) {
      var e = Math.random() * elementMax;
      A.set(i, j, e);
      A.set(j, i, e);
    }
  }
  return A;
};

var createVector = function (n) {
  var elementMax = 100;
  var x = ndarray(new Float64Array(n));
  for (var i = 0; i < n; ++i) {
    x.set(i, Math.random() * elementMax);
  }
  return x;
};

describe('BLAS Level 2', function () {
  var A;
  var A0;
  var x;
  var x0;
  var y;
  // var y0;
  beforeEach(function () {
    var Adata = [1, 2, 5, 3];
    var xdata = [-4, 7];
    var ydata = [3, -2];
    A = ndarray(new Float64Array(Adata), [2, 2]);
    A0 = ndarray(new Float64Array(Adata), [2, 2]);
    x = ndarray(new Float64Array(xdata));
    x0 = ndarray(new Float64Array(xdata));
    y = ndarray(new Float64Array(ydata));
    // y0 = ndarray(new Float64Array(ydata));
  });

  it('gemv', function () {
    assert(blas2.gemv(-4, A, x, 2, y));
    assert.ndCloseTo(ndarray(new Float64Array([-34, -8])), y, 1e-8);
    assert.ndCloseTo(A0, A, 1e-8);
    assert.ndCloseTo(x0, x, 1e-8);
  });

  it('trmv', function () {
    assert(blas2.trmv(A, x));
    assert.ndCloseTo(x, ndarray([10, 21]), 1e-8);
    assert.ndCloseTo(A0, A, 1e-8);
  });

  it('trmv lower', function () {
    assert(blas2.trmv(A, x, true));
    assert.ndCloseTo(x, ndarray([-4, 1]), 1e-8);
    assert.ndCloseTo(A0, A, 1e-8);
  });

  it('trsv', function () {
    assert(blas2.trsv(A, x));
    assert.ndCloseTo(x, ndarray([-8.66666666666666667, 2.3333333333333333]), 1e-8);
    assert.ndCloseTo(A0, A, 1e-8);
  });

  it('trsv lower', function () {
    assert(blas2.trsv(A, x, true));
    assert.ndCloseTo(x, ndarray([-4, 9]), 1e-8);
    assert.ndCloseTo(A0, A, 1e-8);
  });

  it('symv', function () {
    var S = createSymmetricMatrix(10);
    var v = createVector(10);
    var q = ndarray(new Float64Array(10));
    var q0 = ndarray(new Float64Array(10));
    blas2.symv(S, v, q);
    blas2.gemv(1.0, S, v, 0.0, q0);
    assert.ndCloseTo(q, q0, 1e-8);
  });
});
