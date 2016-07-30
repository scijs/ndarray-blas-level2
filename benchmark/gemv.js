var Benchmark = require('benchmark');

var suite = new Benchmark.Suite();
var ndarray = require('ndarray');

var gemvGen = require('../gemv.generic');
var gemvOpt = require('../gemv.optimized');

var m = 50;
var n = 70;
var A = ndarray(new Float64Array(m * n), [m, n]);
var x = ndarray(new Float64Array(n), [n]);
var y = ndarray(new Float64Array(n), [n]);

suite.add('GEMV (get/set notation)', function () {
  gemvGen(1, A, x, 0, y);
})
.add('GEMV (unrolled index arithmetic)', function () {
  gemvOpt(1, A, x, 0, y);
})
.on('cycle', function (event) {
  console.log(String(event.target));
})
.on('complete', function () {
  console.log('Fastest is ' + this.filter('fastest').map('name'));
})
.run({async: false});
