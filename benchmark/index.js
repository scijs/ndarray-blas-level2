var Benchmark = require('benchmark');

var suite = new Benchmark.Suite;
var ndarray = require('ndarray');

var gemvGS = require('./gemv.getset');
var gemvUR = require('./gemv.unrolled');

var m = 50;
var n = 70;
var A = ndarray(new Float64Array(m * n), [m, n]);
var x = ndarray(new Float64Array(n), [n]);
var y = ndarray(new Float64Array(n), [n]);

suite.add('get/set notation', function () {
    gemvGS(1, A, x, 0, y);
  })

  .add('unrolled indices', function () {
    gemvUR(1, A, x, 0, y);
  })

  .on('cycle', function(event) {
    console.log(String(event.target));
  })
  .on('complete', function() {
    console.log('Fastest is ' + this.filter('fastest').map('name'));
  })
  .run({async: false});


