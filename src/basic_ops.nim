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

import math

#### Basic operations

proc `+`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value + rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, 1,
             rhs.index, 1
             )
           )

proc `-`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value - rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, 1,
             rhs.index, -1
             )
           )

proc `*`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value * rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, rhs.value,
             rhs.index, lhs.value
             )
           )

proc `/`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let inv = 1 / rhs.value
  let d = lhs.value * inv
  return Variable[T](
           tape: lhs.tape,
           value: d,
           index: lhs.tape.push_binary(
             lhs.index, inv,
             rhs.index, - d * inv
             )
           )

#### Trigonometric functions

proc cos*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.cos(),
           index: v.tape.push_unary(v.index, -v.value.sin())
           )

proc sin*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.sin(),
           index: v.tape.push_unary(v.index, v.value.cos())
           )

proc tan*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let t = v.value.tan()
  return Variable[T](
           tape: v.tape,
           value: t,
           index: v.tape.push_unary(v.index, 1 + t.pow(2))
           )

proc arccos*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.arccos(),
           index: v.tape.push_unary(v.index, - 1 / sqrt(1 - v.value.pow(2)))
           )

proc arcsin*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.arcsin(),
           index: v.tape.push_unary(v.index, 1 / sqrt(1 - v.value.pow(2)))
           )

proc arctan*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.arctan(),
           index: v.tape.push_unary(v.index, 1 / (1 + v.value.pow(2)))
           )

#### Exponential and logarithms

proc exp*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let e = v.value.exp()
  return Variable[T](
           tape: v.tape,
           value: e,
           index: v.tape.push_unary(v.index, e)
           )

proc ln*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.ln(),
           index: v.tape.push_unary(v.index, 1 / v.value)
           )

proc pow*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let p = lhs.pow(rhs)
  return Variable[T](
           tape: lhs.tape,
           value: p,
           index: lhs.tape.push_binary(
             lhs.index, rhs * lhs.pow(rhs-1),
             rhs.index, p * ln(lhs)
             )
           )

proc log10*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  const ln10 = ln(10)
  return Variable[T](
           tape: v.tape,
           value: v.value.log10(),
           index: v.tape.push_unary(v.index, 1 / (v.value * ln10))
           )

proc sqrt*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let s = v.value.sqrt()
  return Variable[T](
           tape: v.tape,
           value: s,
           index: v.tape.push_unary(v.index, 1 / (2 * s))
           )

#### Hyperbolic functions
## Todo rewrite rules for hyperbolic functions

proc cosh*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.cosh(),
           index: v.tape.push_unary(v.index, v.value.sinh())
           )

proc sinh*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.cosh(),
           index: v.tape.push_unary(v.index, v.value.sinh())
           )

proc tanh*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let t = v.value.tanh()
  return Variable[T](
           tape: v.tape,
           value: t,
           index: v.tape.push_unary(v.index, 1 - t.pow(2))
           )

# acosh, asinh, atanh are not included in Nim Math library
# pending RFC https://github.com/nim-lang/Nim/issues/3745,
# they can be defined in the following manner
# proc atanh(x: float): float {.importc: "atanh", header: "<math.h>".}


#### Basic operations with constants

# constant on the right
proc `+`*[T](lhs: Variable[T], rhs: T): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value + rhs,
           index: lhs.tape.push_unary(lhs.index, 1)
           )

# constant on the left
proc `+`*[T](lhs: T, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: rhs.tape,
           value: lhs + rhs.value,
           index: rhs.tape.push_unary(rhs.index, 1)
           )

# constant on the right
proc `-`*[T](lhs: Variable[T], rhs: T): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value - rhs,
           index: lhs.tape.push_unary(lhs.index, 1)
           )
# constant on the left
proc `-`*[T](lhs: T, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: rhs.tape,
           value: lhs - rhs.value,
           index: rhs.tape.push_unary(rhs.index, -1)
           )

# constant on the right
proc `*`*[T](lhs: Variable[T], rhs: T): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value * rhs,
           index: lhs.tape.push_unary(lhs.index, rhs)
           )

# constant on the left
proc `*`*[T](lhs: T, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: rhs.tape,
           value: lhs * rhs.value,
           index: rhs.tape.push_unary(rhs.index, lhs)
           )

# constant on the right
proc `/`*[T](lhs: Variable[T], rhs: T): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value / rhs,
           index: lhs.tape.push_unary(lhs.index, 1/rhs)
           )

# constant on the left
proc `/`*[T](lhs: T, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let d = lhs / rhs.value
  return Variable[T](
           tape: rhs.tape,
           value: d,
           index: rhs.tape.push_unary(rhs.index, - d / rhs)
           )

# constant on the right
proc pow*[T](lhs: Variable[T], rhs: T): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value.pow(rhs),
           index: lhs.tape.push_unary(lhs.index, rhs * lhs.pow(rhs-1))
           )

# constant on the left
proc pow*[T](lhs: T, rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let p = lhs.pow(rhs)
  return Variable[T](
           tape: rhs.tape,
           value: p,
           index: rhs.tape.push_unary(rhs.index, p * ln(lhs))
           )