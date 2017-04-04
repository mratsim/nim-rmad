# RMAD
Reverse-mode auto-differentiation for Nim. (a.k.a - autograd)

## Goal
Neural networks pass an input through a series of transformations (layers) to get an output value called forward pass.

During the learning phase, neural networks needs to know the contribution of each transformation steps with regards to each input values so its weights can be adjusted. This is the backward propagation pass, a.k.a. backprop, which retrieves the gradients of each transformations with regards to each input values. Those will be used to adjust the weights in the proper direction and a reasonable amplitude.

RMAD, also called autograd is an efficient way to automatically differentiate any transformation applied during the forward pass.

## Examples
Examples are available in the examples folder.

Here is a port of Andrej Karpathy's Javascript old Neural Network [tutorial](https://karpathy.github.io/neuralnets/).

```Nim
let ctx = newContext[float32]()

let
    a = ctx.variable(1)
    b = ctx.variable(2)
    c = ctx.variable(-3)
    x = ctx.variable(-1)
    y = ctx.variable(3)

proc forwardNeuron[T](a,b,c,x,y: T): T =
    let
        ax = a * x
        by = b * y
        axpby = ax + by
        axpbypc = axpby + c
        s = axpbypc.sigmoid()
    return s

var s = forwardNeuron(a,b,c,x,y)

echo s.value() # 0.8807970285415649

let gradient = s.grad()
echo gradient.wrt(a) # -0.1049936264753342
echo gradient.wrt(b) # 0.3149808645248413
echo gradient.wrt(c) # 0.1049936264753342
echo gradient.wrt(x) # 0.1049936264753342
echo gradient.wrt(y) # 0.2099872529506683
```


## Strengths
Everything can be differentiated as long as it can be computed, even a for-loop.
This is a key advantage for Recurrent Neural Networks.
Plus, if you implement your own layer you don't need to derive the function yourself.

## Weaknesses
In place operation of variables (including changing its value) is not supported. As Facebook says, it is a [hard](http://pytorch.org/docs/autograd.html#in-place-operations-on-variables) [matter](https://github.com/pytorch/pytorch/issues/823).

## Status
This library is in its infancy, only +, *, sin, exp and the sigmoid function are implemented. Once every basic ops is done, every function that are based on them (including loop) can be differentiated.

Also matrices support is still pending. It might work with minimal modifications but it's not tested.

As I am developing a ML library based on RMAD, API may (and probably will) change to suit my needs.

## Todo

- [ ] Implement the trigonometric and hyperbolic function from Nim `math` module
- [ ] Approximate equal for floats comparison
- [ ] Add tests
    - basic operations: grad x+x = grad 2x
    - Operator precedence: a + x * y
    - infinity behaviour: grad 1/0
    - trigonometric operations: grad tan x = grad sin x / cos x
    - hyperbolic functions
- [ ] Support matrices and vectors
- [ ] GPU support
- [ ] Support gradient freezing (for transfer learning)

Low priority
- [ ] Support 2nd order and n-th order differentiation
- [ ] Computation graph pretty-printing
- [ ] Canonical form and rewrite rules: 2 / (1 + e^-x) => 2 * x.sigmoid()

## License
Copyright 2017 Mamy André-Ratsimbazafy

The Apache License version 2