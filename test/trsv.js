'use strict';

var chai = require('chai');
var assert = chai.assert;
var RandMatGen = require('./util/rand-matrix-gen.js');
var naiveGEMV = require('./util/naive-gemv');
var ndarray = require('ndarray');
var assertCloseTo = require('./util/close-to');
var constants = require('./util/constants');

var gemv = require('../gemv');
var trsv = require('../trsv');

describe('TRSV (triangular solve)', function () {
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
  var y = ndarray(new Float64Array(m), [m]);
  var y0 = ndarray(new Float64Array(m), [m]);

  it('upper-triangular TRSV', function () {
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, false, B);
      matGen.makeGeneralMatrix(1, n, x);

      assert(gemv(1, B, x, 0, x0));
      assert(trsv(B, x0, false)); // value of x
      assertCloseTo(x, x0, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });

  it('lower-triangular TRSV', function () {
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeTriangularMatrix(n, n, true, B);
      matGen.makeGeneralMatrix(1, n, x);

      assert(gemv(1, B, x, 0, x0));
      assert(trsv(B, x0, true)); // value of x
      assertCloseTo(x, x0, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });
});
