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

type
  # T is float32 or float64, for memory/accuracy tradeoff
  # We do not use Node[T: SomeReal] so that T can be nil in proc newContext[T]
  Node[T] = object
    ## Operation being applied
    weights: array[2, T]
    deps: array[2,int] #ref indices to parent nodes

  Context*[T] = object
    ## Tape / Wengert list. Contains the list of applied operations
    nodes: ref seq[Node[T]]

  Variable*[T] = object
    ## Wrapper for values
    tape: Context[T]
    index: int
    value: T

  Grad[T] = object
    ## Wrapper for the list of gradients with regards to each inputs
    derivs: ref seq[T]

proc newContext*[T]: Context[T] {.noSideEffect.} =
  ## Initialize a context (Tape / Wengert list)
  result.nodes = new seq[Node[T]]
  result.nodes[] = @[]

proc len[T](t: Context[T]): int {.noSideEffect.} =
  ## Returns the number of operations applied in the context
  return t.nodes[].len()

proc push[T](t: Context[T], node: Node[T]) {.noSideEffect.} =
  ## Append a new operation to the context
  t.nodes[].add(node) #Appending in Nim is add not push

proc push_nullary[T](t: Context[T]): int {.noSideEffect.} =
  ## Append a nullary operation to the context
  let len = t.len()
  t.push(
    Node[T](
      weights: [0.T, 0.T],
      deps: [len, len]
      )
    )
  return len

proc push_unary[T](t: Context[T], dep0: int, weight0: T): int {.noSideEffect.} =
  ## Append a unary operation to the context
  let len = t.len()
  t.push(
    Node[T](
      weights: [weight0, 0],
      deps: [dep0, len]
      )
    )
  return len

proc push_binary[T](t: Context[T], dep0: int, weight0: T, dep1: int, weight1: T): int {.noSideEffect.} =
  ## Append a binary operation to the context
  let len = t.len()
  t.push(
    Node[T](
      weights: [weight0, weight1],
      deps: [dep0, dep1]
      )
    )
  return len

proc variable*[T](t: Context[T], value: T): Variable[T] {.noSideEffect.} =
  ## Wrap a variable to the context
  return Variable[T](
           tape: t,
           value: value,
           index: t.push_nullary()
           )

proc value*[T](v: Variable[T]): T {.noSideEffect.} =
  ## Unwrap the value from its context
  return v.value

proc grad*[T](v: Variable[T], pull: T = 1): Grad[T] =
  ## Compute the gradients
  # Computation is done with gradient set to 1 for the final output value
  # If needed it can be set to an arbitrary value (e.g. -1)
  let len = v.tape.len()
  let nodes = v.tape.nodes[]

  result.derivs = new seq[T]

  var derivs = newSeq[T](len)
  derivs[v.index] = pull #by default 1

  for i in countdown(len-1,0):
    let node = nodes[i]
    let deriv = derivs[i]

    for j in 0..1:
      derivs[node.deps[j]] += node.weights[j] * deriv

  result.derivs[] = derivs

proc wrt*[T](g: Grad[T], v: Variable[T]): T {.noSideEffect.} =
  ## Get the gradient with regards to a specific input value
  return g.derivs[v.index]