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

proc sin*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.sin(),
           index: v.tape.push_unary(v.index, v.value.cos())
           )

proc exp*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  return Variable[T](
           tape: v.tape,
           value: v.value.exp(),
           index: v.tape.push_unary(v.index, v.value.exp())
           )