'use strict';

var chai = require('chai');
var assert = chai.assert;
var ndarray = require('ndarray');
var pool = require('ndarray-scratch');
var ops = require('ndarray-ops');
var blas2 = require('../index.js');
var blas1 = require('ndarray-blas-level1');
var RandMatGen = require('./rand-matrix-gen.js');
var show = require('ndarray-show');

var NUM_TESTS = 2000;
var TEST_TOLERANCE = 1e-8;

assert.ndCloseTo = function (a, b, tol, msg) {
  assert(a.dimension === b.dimension, 'expected dimension ' + a.dimension + ' to equal dimension ' + b.dimension + '.');
  assert.deepEqual(a.shape, b.shape, 'expected shape ' + a.shape + ' to equal shape ' + b.shape + '.');
  var c = pool.zeros(a.shape, a.dtype);
  ops.sub(c, a, b);
  var err = ops.norm2(c);
  var errorMsg = msg || '';
  assert(err < tol, 'Expected error ' + err + ' to be less than tolerance ' + tol + '.\n' + errorMsg);
};

var naiveGEMV = function (alpha, A, x, beta, y) {
  var m = A.shape[0];
  var n = A.shape[1];
  for (var i = 0; i < m; ++i) {
    var d = 0;
    for (var j = 0; j < n; ++j) {
      d += A.get(i, j) * x.get(j);
    }
    y.set(i, alpha * d + beta * y.get(i));
  }
  return true;
};

describe('BLAS Level 2', function () {
  var m = 10;
  var n = 15;
  var alpha = 0;
  var beta = 0;
  var seed;
  var matGen = new RandMatGen(seed, Float64Array);
  var A = ndarray(new Float64Array(m * n), [m, n]);
  var B = ndarray(new Float64Array(n * n), [n, n]);
  var x = ndarray(new Float64Array(n), [n]);
  var x0 = ndarray(new Float64Array(n), [n]);
  var xn = ndarray(new Float64Array(n), [n]);
  var y = ndarray(new Float64Array(m), [m]);
  var y0 = ndarray(new Float64Array(m), [m]);

  it('gemv', function () {
    for (var t = 0; t < NUM_TESTS; ++t) {
      alpha = Math.random();
      beta = Math.random();
      seed = matGen.setRandomSeed(36);
      matGen.makeGeneralMatrix(m, n, A);
      matGen.makeGeneralMatrix(1, n, x);

      assert(blas2.gemv(alpha, A, x, beta, y));
      assert(naiveGEMV(alpha, A, x, beta, y0));
      assert.ndCloseTo(y0, y, TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('trmv', function () {
    for (var t = 0; t < NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, false, B);
      matGen.makeGeneralMatrix(1, n, xn);
      blas1.copy(xn, x);

      assert(blas2.trmv(B, x, false));
      assert(blas2.gemv(1, B, xn, 0, x0));
      assert.ndCloseTo(x, x0, TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('trmv lower', function () {
    for (var t = 0; t < NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, true, B);
      matGen.makeGeneralMatrix(1, n, xn);
      blas1.copy(xn, x);

      assert(blas2.trmv(B, x, true));
      assert(blas2.gemv(1, B, xn, 0, x0));
      assert.ndCloseTo(x, x0, TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('trsv', function () {
    for (var t = 0; t < NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, false, B);
      matGen.makeGeneralMatrix(1, n, x);

      assert(blas2.gemv(1, B, x, 0, x0));
      assert(blas2.trsv(B, x0, false)); // value of x
      assert.ndCloseTo(x, x0, TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('trsv lower', function () {
    for (var t = 0; t < NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, true, B);
      matGen.makeGeneralMatrix(1, n, x);

      assert(blas2.gemv(1, B, x, 0, x0));
      assert(blas2.trsv(B, x0, true)); // value of x
      assert.ndCloseTo(x, x0, TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('gbmv', function () {
    matGen.makeBandedMatrix(n, n, 3, 1, B);
    blas2.gbmv(B, 3, 1, x0, x);
    blas2.gemv(1, B, x0, 0, xn);
    assert.ndCloseTo(x, xn, 1e-8);
  });
});
