'use strict';

var chai = require('chai');
var MatrixGenerator = require('./rand-matrix-gen.js');
var printer = require('./debug-printer.js');

describe('Matrix Creation Test', function () {
  it('matrix creation', function () {
    var matGen = new MatrixGenerator('seed string', Float64Array);
    var a = undefined;

    console.log('\n');
    console.log('General Matrix');
    a = matGen.makeGeneralMatrix(5, 5);
    console.log(printer.printMatrix(a));

    console.log('Banded Matrix: 1 lower, 2 upper diagonals');  
    a = matGen.makeBandedMatrix(5, 5, 1, 2);
    console.log(printer.printMatrix(a));

    console.log('Symmetric Matrix');
    a = matGen.makeSymmetricMatrix(5, 5);
    console.log(printer.printMatrix(a));

    console.log('Symmetric Banded Matrix: 2 off-diagonals');
    a = matGen.makeSymmBandedMatrix(5, 2);
    console.log(printer.printMatrix(a));
    
    console.log('Lower Triangular Banded Matrix');
    a = matGen.makeTriangularMatrix(5, 5, true);
    console.log(printer.printMatrix(a));
    
    console.log('Upper Triangular Banded Matrix');
    a = matGen.makeTriangularMatrix(5, 5, false);
    console.log(printer.printMatrix(a));

    console.log('Lower Packed Matrix');
    a = matGen.makePackedMatrix(5);
    console.log(printer.printPackedMatrix(a, true));

    console.log('Upper Packed Matrix');
    console.log(printer.printPackedMatrix(a, false));
  });
});