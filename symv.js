'use strict';

var blas1 = require('ndarray-blas-level1');

exports.symv = function (A, x, y, fromLower, alpha, beta) {
  var scal = blas1.scal;
  var n = A.shape[0];

  var lower = fromLower || true;
  var alpha0 = alpha || 1.0;
  var beta0 = beta || 0.0;

  var i = 0;
  var j = 0;
  var t1 = 0;
  var t2 = 0;
  scal(beta0, y);
  if (lower) {
    for (j = 0; j < n; ++j) {
      t1 = alpha0 * x.get(j);
      t2 = 0;
      y.set(j, y.get(j) + t1 * A.get(j, j));
      for (i = j + 1; i < n; ++i) {
        y.set(i, y.get(i) + t1 * A.get(i, j));
        t2 = t2 + A.get(i, j) * x.get(i);
      }
      y.set(j, y.get(j) + alpha0 * t2);
    }
  } else {
    for (j = 0; j < n; ++j) {
      t1 = alpha0 * x.get(j);
      t2 = 0;
      for (i = 0; i <= j - 1; ++i) {
        y.set(i, y.get(i) + t1 * A.get(i, j));
        t2 = t2 + A.get(i, j) * x.get(i);
      }
      y.set(j, y.get(j) + t1 * A.get(j, j) + alpha0 * t2);
    }
  }
  return true;
};
