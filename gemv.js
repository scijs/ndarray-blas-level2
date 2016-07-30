'use strict';

module.exports = gemv;

var gemvGeneric = require('./gemv.generic.js');
var gemvOpt = require('./gemv.optimized.js');

function gemv (alpha, A, x, beta, y) {
  if (A.dtype === 'generic' || x.dtype === 'generic' || y.dtype === 'generic') {
    return gemvGeneric(alpha, A, x, beta, y);
  } else {
    return gemvOpt(alpha, A, x, beta, y);
  }
}
