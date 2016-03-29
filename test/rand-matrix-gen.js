'use strict';

var ndarray = require('ndarray');
var ops = require('ndarray-ops');
var ndfill = require('ndarray-fill');
var ndband = require('ndarray-band');
var pool = require('ndarray-scratch');
var mersennetwister = require('mersennetwister');

module.exports = function (seed, arrayType) {
  var NUMBER_ARRAY = arrayType;
  var randGenSeed = seed;
  var prng = new mersennetwister(seed);

  var exports = {};

  exports.getSeed = function () {
    if(!prng) {
      throw new Error('Number generator not initialized.');
    }
    return randGenSeed;
  }
  exports.setNewSeed = function (seed) {
    if(!prng) {
      throw new Error('Number generator not initialized.');
    }
    prng.seed(seed);
    randGenSeed = seed;
  };
  exports.setRandomSeed = function (length) {
    if(!prng) {
      throw new Error('Number generator not initialized.');
    }
    var len = length || 36;
    var text = "";
    var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for (var i = 0; i < len; ++i) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    prng.seed(text);
    randGenSeed = text;
    return text;
  };
  exports.makeGeneralMatrix = function (m, n, M) {
    if(!prng) {
      throw new Error('Number generator not initialized.');
    }
    var numElems = m * n;

    var A = M || ndarray(new NUMBER_ARRAY(numElems), [m, n]);
    ndfill(A, function(i, j) {
      return prng.random();
    });
    return A;
  };
  exports.makeSymmetricMatrix = function(n, M) {
    if(!prng) {
      throw new Error('Number generator not initialized.');
    }
    var numElems = n * n;
    var A = M || ndarray(new NUMBER_ARRAY(numElems), [n, n]);
    ops.assigns(A, 0);
    ndfill(A, function(i, j) {
      var val = A.get(j, i);
      if(val !== 0) {
        return val;
      } else {
        return prng.random();  
      }
    });
    return A;
  };
  exports.makeSymmBandedMatrix = function(n, k, M) {
    if(!prng) {
      throw new Error('Number generator not initialized.');
    }
    var numElems = n * n;
    var A = M || ndarray(new NUMBER_ARRAY(numElems), [n, n]);
    ops.assigns(A, 0);
    
    var band1 = ndband(A, 0);
    for (var j = 0; j < band1.shape[0]; ++j) {
      band1.set(j, prng.random());
    }
    for (var i = 1; i <= k; ++i) {
      band1 = ndband(A, i);
      var band2 = ndband(A, -i);
      for (var j = 0; j < band1.shape[0]; ++j) {
        var val = prng.random();
        band1.set(j, val);
        band2.set(j, val);
      }
    }
    return A;
  };

  exports.makeBandedMatrix = function (m, n, kl, ku, M) {
    if(!prng) {
      throw new Error('Number generator not initialized.');
    }
    var numElems = m * n;
    var A = M || ndarray(new NUMBER_ARRAY(numElems), [m, n]);
    var numLower = Math.min(kl, m);
    var numUpper = Math.min(ku, n);
    
    for (var i = 0; i <= numLower; ++i) {
      var band = ndband(A, i);
      for (var j = 0; j < band.shape[0]; ++j) {
        band.set(j, prng.random());
      }
    }
    for (var i = 0; i <= numUpper; ++i) {
      var band = ndband(A, -i);
      for (var j = 0; j < band.shape[0]; ++j) {
        band.set(j, prng.random());
      }
    }    
    return A;
  };
  exports.makeTriangularMatrix = function (m, n, lower, M) {
    if(!prng) {
      throw new Error('Number generator not initialized.');
    }
    var numElems = m * n;
    var A = M || ndarray(new NUMBER_ARRAY(numElems), [m, n]);
    ops.assigns(A, 0);

    if(lower) {
      for (var i = 0; i < m; ++i) {
        var band = ndband(A, i);
        for (var j = 0; j < band.shape[0]; ++j) {
          band.set(j, prng.random());
        }
      }
    } else {
      for (var i = 0; i < n; ++i) {
        var band = ndband(A, -i);
        for (var j = 0; j < band.shape[0]; ++j) {
          band.set(j, prng.random());
        }
      }
    }
    return A;
  };
  exports.makePackedMatrix = function (n, M) {
    var numElems = ((n + 1) * n)/ 2;
    var A = M || ndarray(new NUMBER_ARRAY(numElems), [numElems]);
    for (var j = 0; j < numElems; ++j) {
      A.set(j, prng.random());
    }  
    return A;
  }

  return exports;
};
