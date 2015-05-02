'use strict';

var blas1 = require('ndarray-blas-level1');

exports.gemv = function(alpha, A, x, beta, y) {
  var dot = blas1.dot;
  for(var i=A.shape[1]-1; i>=0; i--) {
    y.set(i, y.get(i)*beta + alpha * dot( A.pick(i,null), x));
  }
  return true;
};

exports.gbmv = function() {
  console.error('GBMV (banded matrix vector multiply) not yet implemented');
};

exports.symv = function() {
  console.error('SYMV (symmetric matrix vector multiply) not yet implemented');
};

exports.sbmv = function() {
  console.error('SBMV (symmetric banded matrix vector multiply) not yet implemented');
};

exports.spmv = function() {
  console.error('SPMV (symmetric packed matrix vector multiply) not yet implemented');
};


// Compute the product of an upper triangular matrix with a vector
exports.trmv = function(A, x, isLower) {
  var dot = blas1.dot;
  var n = A.shape[1];
  if( isLower ) {
    for(var i=n-1; i>=0; i--) {
      x.set(i, dot( A.pick(i,null).hi(i+1), x.hi(i+1) ) );
    }
  } else {
    for(var i=0; i<n; i++) {
      x.set(i, dot( A.pick(i,null).lo(i), x.lo(i) ) );
    }
  }
  return true;
};

exports.trmv_lower = function(A,x) {
  console.warn('trmv_lower is deprected. Please use the \'isLower\' flag with trmv');
  return exports.trmv(A,x,true);
}

exports.tbmv = function() {
  console.error('TBMV (triangular banded matrix vector multiply) not yet implemented');
};

// Solve Ax=b where A is upper triangular
exports.trsv = function(A, x, isLower) {
  var dot = blas1.dot;
  var n = A.shape[1];
  if( isLower ) {
    x.set( 0, x.get(0)/A.get(0,0) );
    for(var i=1; i<n; i++) {
      x.set(i, (x.get(i) - dot(A.pick(i,null).hi(i), x.hi(i))) / A.get(i,i) );
    }
  } else {
    x.set( n-1, x.get(n-1)/A.get(n-1,n-1) );
    for(var i=n-2; i>=0; i--) {
      x.set(i, (x.get(i) - dot(A.pick(i,null).lo(i+1), x.lo(i+1))) / A.get(i,i) );
    }
  }
  return true;
};

// Solve Ax=b where A is lower triangular
exports.trsv_lower = function(A, x) {
  console.warn('trsv_lower is deprected. Please use the \'isLower\' flag with trsv');
  return exports.trsv(A,x,true);
};

exports.tbsv = function() {
  console.error('TBSV (triangular banded matrix solver) not yet implemented');
};

exports.tpsv = function() {
  console.error('TPSV (triangular packed matrix solver) not yet implemented');
};

exports.ger = function() {
  console.error('GER (rank 1 operation A := alpha*x*y\' + A) not yet implemented');
};

exports.syr = function() {
  console.error('SYR (symmetric rank 1 operation A := alpha*x*y\' + A) not yet implemented');
};

exports.spr = function() {
  console.error('SPR (symmetric packed rank 1 operation A := alpha*x*y\' + A) not yet implemented');
};

exports.syr2 = function() {
  console.error('SYR (symmetric rank 2 operation A := alpha*x*y\' + alpha*y*x\' + A) not yet implemented');
};

exports.spr2 = function() {
  console.error('SPR (symmetric packed rank 2 operation A := alpha*x*y\' + alpha*y*x\' + A) not yet implemented');
};
