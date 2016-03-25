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

describe('BLAS Level 2', function () {
  var A;
  var A0;
  var x;
  var x0;
  var y;
  // var y0;
  var B;
  var z;
  var w;
  var w0;
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

    var Bdata = [ 85.04423347767442,  9.361860854551196,  0,  0,  0,  0,  0,  0,
              82.14579129125923,  50.819494598545134, 45.74480822775513,  0,  0,  0,  0,  0,
              0.19750702194869518,  9.582619182765484,  1.4352485537528992, 12.11472856812179,  0,  0,  0,  0,
              58.657241822220385, 30.14892488718033,  16.522647975943983, 18.874867935664952, 81.8313357187435, 0,  0,  0,
              0,  81.81838407181203,  76.312301075086,  76.66437732987106,  88.07732549030334,  48.417096375487745, 0,  0,
              0,  0,  32.64176605734974,  73.48592570051551,  96.89196601975709,  24.069529958069324, 43.76116660423577,  0];
    var zdata = [45.638499688357115,46.46283083129674,3.4731490770354867,73.26457547023892,45.81439339090139,61.162385856732726,22.422668361105025,20.822709542699158];
    B = ndarray(new Float64Array(Bdata), [6,8]);
    z = ndarray(new Float64Array(zdata));
    w = ndarray(new Float64Array([0,0,0,0,0,0]));
    w0 = ndarray(new Float64Array([0,0,0,0,0,0]));
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

  it('gbmv', function () {
    blas2.gbmv(B, 3, 1, z, w);
    blas2.gemv(1.0, B, z, 0.0, w0);
    assert.ndCloseTo(w, w0, 1e-8);
  });
});
