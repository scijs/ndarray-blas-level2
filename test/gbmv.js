'use strict';

var chai = require('chai');
var assert = chai.assert;
var RandMatGen = require('./util/rand-matrix-gen.js');
var naiveGEMV = require('./util/naive-gemv');
var ndarray = require('ndarray');
var assertCloseTo = require('./util/close-to');
var constants = require('./util/constants');

var gemv = require('../gemv');
var gbmv = require('../gbmv');

describe('GBMV (general banded matrix-vector product)', function () {
  var m = 10;
  var n = 15;
  var alpha = 0;
  var beta = 0;
  var seed;
  var matGen = new RandMatGen(seed, Float64Array);
  var A = ndarray(new Float64Array(m * n), [m, n]);
  var x = ndarray(new Float64Array(n), [n]);
  var x0 = ndarray(new Float64Array(n), [n]);
  var xn = ndarray(new Float64Array(n), [n]);
  var y = ndarray(new Float64Array(m), [m]);
  var y0 = ndarray(new Float64Array(m), [m]);
  var B = ndarray(new Float64Array(n * n), [n, n]);

  it('gbmv', function () {
    for (var t = 0; t < constants.NUM_TESTS; ++t) {
      seed = matGen.setRandomSeed(36);
      matGen.makeBandedMatrix(n, n, 3, 1, B);
      gbmv(B, 3, 1, x0, x);
      gemv(1, B, x0, 0, xn);
      assertCloseTo(x, xn, constants.TEST_TOLERANCE, 'Failure seed value: "' + seed + '".');
    }
  });
});
