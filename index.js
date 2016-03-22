'use strict';

module.exports.gemv = require('./gemv.js').gemv;
module.exports.gbmv = require('./gbmv.js').gbmv;
module.exports.symv = require('./symv.js').symv;
module.exports.sbmv = require('./sbmv.js').sbmv;
module.exports.spmv = require('./spmv.js').spmv;
module.exports.trmv = require('./trmv.js').trmv;
module.exports.tbmv = require('./tbmv.js').tbmv;
module.exports.trsv = require('./trsv.js').trsv;
module.exports.tbsv = require('./tbsv.js').tbsv;
module.exports.tpsv = require('./tpsv.js').tpsv;
module.exports.ger = require('./ger.js').ger;
module.exports.syr = require('./syr.js').syr;
module.exports.spr = require('./spr.js').spr;
module.exports.syr2 = require('./syr2.js').syr2;
module.exports.spr2 = require('./spr2.js').spr2;
module.exports.trmv_lower = function (A, x) {
  console.warn('trmv_lower is deprecated. Please use the \'isLower\' flag with trmv.');
  return module.exports.trmv(A, x, true);
};
module.exports.trsv_lower = function (A, x) {
  console.warn('trsv_lower is deprecated. Please use the \'isLower\' flag with trsv.');
  return module.exports.trsv(A, x, true);
};
