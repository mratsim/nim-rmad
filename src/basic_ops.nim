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

proc `+`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  #TODO Check that both Variable use the same tape/context
  #TODO Check gradient of a + a
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value + rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, 1,
             rhs.index, 1
             )
           )

proc `-`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  #TODO Check that both Variable use the same tape/context
  #TODO Check gradient of a + a
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value + rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, 1,
             rhs.index, -1
             )
           )

proc `*`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  #TODO Check that both Variable use the same tape/context
  #TODO Check gradient of a * a
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value * rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, rhs.value,
             rhs.index, lhs.value
             )
           )

proc `/`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  #TODO Check that both Variable use the same tape/context
  #TODO Check gradient of a * a
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value * rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, rhs.value,
             rhs.index, -lhs.value / (rhs.value.pow(2))
             )
           )

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