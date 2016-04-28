'use strict';

module.exports = gemv;

function gemv (alpha, A, x, beta, y) {
  var val;
  var adata = A.data;
  var ao = A.offset;
  var as0 = A.stride[0];
  var as1 = A.stride[1];
  var xdata = x.data;
  var xo = x.offset;
  var xs = x.stride[0];
  var ydata = y.data;
  var yo = x.offset;
  var ys = y.stride[0];
  for (var i = A.shape[0] - 1; i >= 0; --i) {
    val = 0;
    for (var j = A.shape[1] - 1; j >= 0; --j) {
      val += adata[ao + as0 * i + as1 * j] * xdata[xo + xs * j];
    }
    ydata[yo + ys * i] = ydata[yo + ys * i] * beta + alpha * val;
  }
  return true;
}
