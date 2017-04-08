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



#######################################
## Documentation on the data structure:

# To store all operations on `Variable` we have a context, which is mathematically a tape/Wengert list.
# In the program it's a sequence of `Nodes`
#
# Each `Nodes` represent an operation f(a, b) and stores:
# - in weights field, functions to differentiate w.r.t. a and b: df/da and df/db
#   we don't store the result we store the actual function for https://github.com/mratsim/nim-rmad/issues/2
# - in parents field, we store the index of the parent `Node` (function) of the operand.
#   Nullary and Unary function are given a dummy length index.
#
# `Variable` contains a copy (not a reference) of the context that created them.
# They also have their index within that Context, to be used by the function wrt and store their value.
#
# `Grad` is just a wrapper for a list of gradients. Each operations/variables knows the proper index to retrieve

## TODO
# Profile the memory of `Variable` storing `Context` by value.
# Note: Compiler should optimize `Context` to just a pointer anyway
#
# 
#######################################

type
  # T is float32 or float64, for memory/accuracy tradeoff
  # We do not use Node[T: SomeReal] so that T can be nil in proc newContext[T]

  # To ease search, backward propagation procedures are prefixed with bp_
  BackProp[T] = proc (gradient: T): T {.noSideEffect.}

  Node[T] = object
    ## Represent an operation
    ## Stores the gradient transformation for backprop in weights
    ## Stores indices of parent operation in parents
    weights: array[2, BackProp[T]]
    parents: array[2,int] #ref indices to parent nodes

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

# Templates in Nim are always inlined. They are used for performance reason to save on function calls costs.

proc newContext*[T]: Context[T] {.noSideEffect.} =
  ## Initialize a context (Tape / Wengert list)
  result.nodes = new seq[Node[T]]
  result.nodes[] = @[]

template len[T](t: Context[T]): int =
  ## Returns the number of operations applied in the context
  t.nodes[].len()

template push[T](t: Context[T], node: Node[T]) =
  ## Append a new operation to the context
  t.nodes[].add(node) #Appending in Nim is add not push

proc push_nullary[T](t: Context[T]): int {.noSideEffect.} =
  ## Append a nullary operation to the context
  let len = t.len()
  proc bp_0[T](gradient: T): T {.noSideEffect, closure.}= 0

  t.push(
    Node[T](
      weights: [bp_0[T], bp_0[T]],
      parents: [len, len]
      )
    )
  return len

proc push_unary[T](t: Context[T], parent0: int, weight0: BackProp[T]): int {.noSideEffect.} =
  ## Append a unary operation to the context
  let len = t.len()
  proc bp_0[T](gradient: T): T {.noSideEffect, closure.}= 0

  t.push(
    Node[T](
      weights: [weight0, bp_0[T]],
      parents: [parent0, len]
      )
    )
  return len

proc push_binary[T](t: Context[T], parent0: int, weight0: BackProp[T], parent1: int, weight1: BackProp[T]): int {.noSideEffect.} =
  ## Append a binary operation to the context
  let len = t.len()
  t.push(
    Node[T](
      weights: [weight0, weight1],
      parents: [parent0, parent1]
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

template value*[T](v: Variable[T]): T  =
  ## Unwrap the value from its context
  v.value

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
      derivs[node.parents[j]] += node.weights[j](deriv)

  result.derivs[] = derivs

template wrt*[T](g: Grad[T], v: Variable[T]): T =
  ## Get the gradient with regards to a specific input value
  g.derivs[v.index]