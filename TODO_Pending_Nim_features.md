To be implemented in the best way, some features would need some issues to be ironed out:

 1. Prevent operation on different contexts
 
What is possible today:

```Nim
Context*[T] = object
  ## Tape / Wengert list. Contains the list of applied operations
  nodes: ref seq[Node[T]]
  context_id: int
```
And compare context_id for each operations I'm implementing
This is expensive especially in a loop (for Recurrent Neural Networks) and prone to mistakes (forget the if/then implementation on new operations)

The best way would be to have Nim typechecker check the Context at compile-time and avoid the costs at runtime

One way to do this is by having a global counter, scoped to the module to avoid name conflict.
This is pending https://github.com/nim-lang/Nim/issues/3845: 

```Nim
type
  Node[T] = object

  Context*[T; I: static[int]] = object
    nodes: ref seq[Node[T]]

var CONTEXT_COUNTER {.compiletime.} = 0

proc ctx_handler(T: typedesc, I: static[int]): Context[T, I] {.noSideEffect.} =
  result.nodes = new seq[Node[T]]
  result.nodes[] = @[]

proc newContext*(T: typedesc): auto =
  inc CONTEXT_COUNTER
  return ctx_handler(T, CONTEXT_COUNTER)

let ctx = newContext(float32) # This line doesn't compile
```


The other way would be to have context as a generic parameter to variable proc and Variable object.
That would require "generic generic parameters" https://github.com/nim-lang/Nim/issues/3856

```Nim
type
  Node[T] = object

  Context*[T] = object
    nodes: ref seq[Node[T]]
  
  Variable*[T, CTX: Context] = object
    ## Wrapper for values
    tape: CTX # Generic generic would be lively to add the T constraint: https://github.com/nim-lang/Nim/issues/3856
    index: int
    value: T

proc newContext*[T]: Context[T] {.noSideEffect.} =
  ## Initialize a context (Tape / Wengert list)
  result.nodes = new seq[Node[T]]
  result.nodes[] = @[]

let ctx = newContext[float32]

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

proc variable*[T](CTX: Context, value: T): Variable[T, CTX] {.noSideEffect.} =
  ## Wrap a variable to the context
  return Variable[T, CTX](
           tape: t,
           value: value,
           index: t.push_nullary()
           )
```