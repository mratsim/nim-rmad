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

proc sigmoid*[T](v: Variable[T]): Variable[T] {.noSideEffect.} =
  let s = 1 / (1 + exp(-v.value))
  return Variable[T](
           tape: v.tape,
           value: s,
           index: v.tape.push_unary(v.index, s * (1-s))
           )