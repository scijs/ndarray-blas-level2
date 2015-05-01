'use strict';

var blas1 = require('ndarray-blas-level1'),
    pool = require('ndarray-scratch');

exports.gemv = function(alpha, A, x, beta, y) {
  var dot = blas1.dot;
  for(var i=A.shape[1]-1; i>=0; i--) {
    y.set(i, y.get(i)*beta + alpha * dot( A.pick(i,null), x));
  }
};

exports.trmv = function(A, x) {
  var dot = blas1.dot;
  var n = A.shape[1];
  for(var i=0; i<n; i++) {
    x.set(i, dot( A.pick(i,null).lo(i), x.lo(i) ) );
  }
};

exports.trsv = function(A, x) {
  var dot = blas1.dot;
  var n = A.shape[1];
  x.set( n-1, x.get(n-1)/A.get(n-1,n-1) );
  for(var i=n-2; i>=0; i--) {
    x.set(i, (x.get(i) - dot(A.pick(i,null).lo(i+1), x.lo(i+1))) / A.get(i,i) );
  }
};
