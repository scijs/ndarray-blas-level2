'use strict';

var chai = require('chai');
var assert = chai.assert;
var ndarray = require('ndarray');
var blas2 = require('../index.js');
var blas1 = require('ndarray-blas-level1');
var RandMatGen = require('./rand-matrix-gen.js');
var naiveGEMV = require('./naive-gemv');
var assertCloseTo = require('./close-to');
var constants = require('./constants');
// var show = require('ndarray-show');

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
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      alpha = Math.random();
      beta = Math.random();
      seed = matGen.setRandomSeed(36);
      matGen.makeGeneralMatrix(m, n, A);
      matGen.makeGeneralMatrix(1, n, x);

      assert(blas2.gemv(alpha, A, x, beta, y));
      assert(naiveGEMV(alpha, A, x, beta, y0));
      assertCloseTo(y0, y, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('trmv', function () {
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, false, B);
      matGen.makeGeneralMatrix(1, n, xn);
      blas1.copy(xn, x);

      assert(blas2.trmv(B, x, false));
      assert(blas2.gemv(1, B, xn, 0, x0));
      assertCloseTo(x, x0, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('trmv lower', function () {
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, true, B);
      matGen.makeGeneralMatrix(1, n, xn);
      blas1.copy(xn, x);

      assert(blas2.trmv(B, x, true));
      assert(blas2.gemv(1, B, xn, 0, x0));
      assertCloseTo(x, x0, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('trsv', function () {
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, false, B);
      matGen.makeGeneralMatrix(1, n, x);

      assert(blas2.gemv(1, B, x, 0, x0));
      assert(blas2.trsv(B, x0, false)); // value of x
      assertCloseTo(x, x0, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('trsv lower', function () {
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, true, B);
      matGen.makeGeneralMatrix(1, n, x);

      assert(blas2.gemv(1, B, x, 0, x0));
      assert(blas2.trsv(B, x0, true)); // value of x
      assertCloseTo(x, x0, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('gbmv', function () {
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeBandedMatrix(n, n, 3, 1, B);
      blas2.gbmv(B, 3, 1, x0, x);
      blas2.gemv(1, B, x0, 0, xn);
      assertCloseTo(x, xn, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });
});
