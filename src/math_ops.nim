# Copyright 2017 Mamy André-Ratsimbazafy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##########################################################################
###### Basic operations from Nim math library

import math, future

#### Basic operations

## Basic gradient transformations for backward pass
proc bp_identity[T](gradient: T): T {.noSideEffect.}= gradient
proc bp_negate[T](gradient: T): T {.noSideEffect.}= -gradient

proc `+`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value + rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, bp_identity[T],
             rhs.index, bp_identity[T]
             )
           )

proc `-`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value - rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, bp_identity[T],
             rhs.index, bp_negate[T]
             )
           )

# Multiplication
template bp_mul[T](hs: T): BackProp[T] =
  (gradient: T) => gradient * hs

proc `*`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value * rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, bp_mul[T](rhs.value),
             rhs.index, bp_mul[T](lhs.value)
             )
           )

# Division
template bp_inverse_rhs[T](inv: T): BackProp[T] =
  (gradient: T) => gradient * inv
template bp_minusdiv_lhs_2[T](d, rhs: T): BackProp[T] =
  (gradient: T) => gradient * -d / rhs

proc `/`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let inv = 1 / rhs.value
  let d = lhs.value / rhs.value # We could multiply by `inv` but we lose accuracy fast ! (mul is 5x faster than div)

  return Variable[T](
           tape: lhs.tape,
           value: d,
           index: lhs.tape.push_binary(
             lhs.index, bp_inverse_rhs(inv),
             rhs.index, bp_minusdiv_lhs_2(d,rhs.value)
             )
           )

#### Trigonometric functions
template bp_negate_sin[T](value: T): BackProp[T] =
  (gradient: T) => - gradient * value.sin()

template bp_cos[T](value: T): BackProp[T] =
  (gradient: T) => gradient * value.cos()

template bp_1ptan2[T](t: T): BackProp[T] =
  (gradient: T) => gradient * (1 + t * t)

template bp_diff_arccos[T](value: T): BackProp[T] =
  (gradient: T) => - gradient / sqrt(1 - value * value)

template bp_diff_arcsin[T](value: T): BackProp[T] =
  (gradient: T) => gradient / sqrt(1 - value * value)

template bp_diff_arctan[T](value: T): BackProp[T] =
  (gradient: T) => gradient / (1 + value * value)

proc cos*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.cos(),
           index: v.tape.push_unary(v.index, bp_negate_sin(v.value))
           )

proc sin*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.sin(),
           index: v.tape.push_unary(v.index, bp_cos(v.value))
           )

proc tan*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let t = v.value.tan()
  return Variable[T](
           tape: v.tape,
           value: t,
           index: v.tape.push_unary(v.index, bp_1ptan2(t))
           )

proc arccos*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.arccos(),
           index: v.tape.push_unary(v.index, bp_diff_arccos(v.value))
           )

proc arcsin*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.arcsin(),
           index: v.tape.push_unary(v.index, bp_diff_arcsin(v.value))
           )

proc arctan*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.arctan(),
           index: v.tape.push_unary(v.index, bp_diff_arctan(v.value))
           )

#### Exponential and logarithms
template bp_exp[T](e: T): BackProp[T] =
  (gradient: T) => gradient * e

template bp_inverse[T](value: T): BackProp[T] =
  (gradient: T) => gradient / value

template bp_diff_pow_lhs[T](lhs, rhs: T): BackProp[T] =
  (gradient: T) => gradient * rhs * lhs.pow(rhs-1)

template bp_diff_pow_rhs[T](p, lhs: T): BackProp[T] =
  (gradient: T) => gradient * p * ln(lhs)

template bp_diff_log10[T](value, ln10: T): BackProp[T] =
  (gradient: T) => gradient / (value * ln10)

template bp_diff_sqrt[T](s: T): BackProp[T] =
  (gradient: T) => gradient / (2 * s)

proc exp*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let e = v.value.exp()
  return Variable[T](
           tape: v.tape,
           value: e,
           index: v.tape.push_unary(v.index, bp_exp(e))
           )

proc ln*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.ln(),
           index: v.tape.push_unary(v.index, bp_inverse(v.value))
           )

proc pow*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let p = lhs.value.pow(rhs.value)
  return Variable[T](
           tape: lhs.tape,
           value: p,
           index: lhs.tape.push_binary(
             lhs.index, bp_diff_pow_lhs(lhs.value, rhs.value),
             rhs.index, bp_diff_pow_rhs(p, lhs.value)
             )
           )

proc log10*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  const ln10 = ln(10.T)
  return Variable[T](
           tape: v.tape,
           value: v.value.log10(),
           index: v.tape.push_unary(v.index, bp_diff_log10(v.value,ln10))
           )

proc sqrt*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let s = v.value.sqrt()
  return Variable[T](
           tape: v.tape,
           value: s,
           index: v.tape.push_unary(v.index, bp_diff_sqrt(s))
           )

#### Hyperbolic functions
## Todo rewrite rules for hyperbolic functions
template bp_sinh[T](value: T): BackProp[T] =
  (gradient: T) => gradient * value.sinh()

template bp_cosh[T](value: T): BackProp[T] =
  (gradient: T) => gradient * value.cosh()

template bp_diff_tanh[T](t: T): BackProp[T] =
  (gradient: T) => gradient * (1 - t * t)

proc cosh*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.cosh(),
           index: v.tape.push_unary(v.index, bp_sinh(v.value))
           )

proc sinh*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.cosh(),
           index: v.tape.push_unary(v.index, bp_cosh(v.value))
           )

proc tanh*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let t = v.value.tanh()
  return Variable[T](
           tape: v.tape,
           value: t,
           index: v.tape.push_unary(v.index, bp_diff_tanh(t))
           )

# acosh, asinh, atanh are not included in Nim Math library
# pending RFC https://github.com/nim-lang/Nim/issues/3745,
# they can be defined in the following manner
# proc atanh(x: float): float {.importc: "atanh", header: "<math.h>".}


#### Basic operations with constants
## TODO: profiling all the type conversion that happens

# constant on the right
proc `+`*[T;U: SomeNumber](lhs: Variable[T], rhs: U): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value + rhs.T,
           index: lhs.tape.push_unary(lhs.index, bp_identity[T])
           )

# constant on the left
proc `+`*[T;U: SomeNumber](lhs: U, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: rhs.tape,
           value: lhs.T + rhs.value,
           index: rhs.tape.push_unary(rhs.index, bp_identity[T])
           )

# constant on the right
proc `-`*[T;U: SomeNumber](lhs: Variable[T], rhs: U): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value - rhs.T,
           index: lhs.tape.push_unary(lhs.index, bp_identity[T])
           )
# constant on the left
proc `-`*[T;U: SomeNumber](lhs: U, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: rhs.tape,
           value: lhs.T - rhs.value,
           index: rhs.tape.push_unary(rhs.index, bp_negate[T])
           )

# constant on the right
proc `*`*[T;U: SomeNumber](lhs: Variable[T], rhs: U): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value * rhs.T,
           index: lhs.tape.push_unary(lhs.index, bp_mul(rhs.T))
           )

# constant on the left
proc `*`*[T;U: SomeNumber](lhs: U, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: rhs.tape,
           value: lhs.T * rhs.value,
           index: rhs.tape.push_unary(rhs.index, bp_mul(lhs.T))
           )

# constant on the right
proc `/`*[T;U: SomeNumber](lhs: Variable[T], rhs: U): Variable[T] {.noSideEffect.} =
  let inv = 1 / rhs.T
  let d = lhs.value / rhs.T # We could multiply by `inv` but we lose accuracy fast ! (mul is 5x faster than div)

  return Variable[T](
           tape: lhs.tape,
           value: d,
           index: lhs.tape.push_unary(lhs.index, bp_inverse_rhs(inv))
           )

# constant on the left
proc `/`*[T;U: SomeNumber](lhs: U, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let d = lhs.T / rhs.value
  return Variable[T](
           tape: rhs.tape,
           value: d,
           index: rhs.tape.push_unary(rhs.index, bp_minusdiv_lhs_2(d,rhs.value))
           )

# constant on the right
proc pow*[T;U: SomeNumber](lhs: Variable[T], rhs: U): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value.pow(rhs.T),
           index: lhs.tape.push_unary(lhs.index, bp_diff_pow_lhs(lhs.value, rhs.T))
           )

# constant on the left
proc pow*[T;U: SomeNumber](lhs: U, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let p = lhs.T.pow(rhs.value)
  return Variable[T](
           tape: rhs.tape,
           value: p,
           index: rhs.tape.push_unary(rhs.index, bp_diff_pow_rhs(p, lhs.T))
           )