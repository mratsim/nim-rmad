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

template bp_passedthrough[T](m, test: T): BackProp[T] =
  (gradient: T) => (if test == m: gradient else: 0)

proc max*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let m: T = max(lhs.value, rhs.value)
  return Variable[T](
           tape: lhs.tape,
           value: m,
           index: lhs.tape.push_binary(
             lhs.index, bp_passedthrough(m, lhs.value),
             rhs.index, bp_passedthrough(m, rhs.value)
             )
           )

proc min*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] {.noSideEffect.} =
  let m: T = min(lhs.value, rhs.value)
  return Variable[T](
           tape: lhs.tape,
           value: m,
           index: lhs.tape.push_binary(
             lhs.index, bp_passedthrough(m, lhs.value),
             rhs.index, bp_passedthrough(m, rhs.value)
             )
           )

template bp_sigmoid[T](sigmoid: T): BackProp[T] =
  (gradient: T) => sigmoid * (1 - sigmoid)

proc sigmoid*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let s = 1 / (1 + exp(-v.value))
  return Variable[T](
           tape: v.tape,
           value: s,
           index: v.tape.push_unary(v.index, bp_sigmoid(s))
           )

proc relu*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let r: T = max(v.value, 0)
  return Variable[T](
           tape: v.tape,
           value: r,
           index: v.tape.push_unary(v.index, bp_passedthrough(r, v.value))
           )