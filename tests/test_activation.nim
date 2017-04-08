import ../rmad.nim, unittest, math

# Approximate Floating point comparison
# Considerations https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
# Implementation of a general solution is probably best on R
# https://github.com/PredictiveEcology/fpCompare
proc `=~`[T:SomeReal, U:SomeNumber](x, y: T|U): bool = (abs(x.T-y.T) < 1e-08)

suite "Activation functions autodifferentiation":
    test "Approximate comparison sanity check":
        check: not (1'f32 =~ 1.0000001)
        check: 1'f32 =~ 1.00000001

    test "Basic operations - Maximum":
        let ctx = newContext[float32]()
        let a = ctx.variable(10)
        let b = ctx.variable(5)

        let k = max(a,b)
        check: k.value =~ 10'f32
        check: k.grad.wrt(a) =~ 1'f32
        check: k.grad.wrt(b) =~ 0'f32

    test "Basic operations - Minimum":
        let ctx = newContext[float32]()
        let a = ctx.variable(10)
        let b = ctx.variable(-5)

        let k = min(a,b)
        check: k.value =~ -5'f32
        check: k.grad.wrt(a) =~ 0'f32
        check: k.grad.wrt(b) =~ 1'f32

    test "Activation - Sigmoid":
        let ctx = newContext[float32]()
        let a = ctx.variable(-2)
        let b = ctx.variable(0)

        let sb = sigmoid(b)
        let integral = ln(1 + exp(a))

        check: integral.grad.wrt(a) =~ sigmoid(a).value
        check: sb.grad.wrt(b) - (sb * (1 - sb)).value =~ 0

    test "Activation - ReLU":
        let ctx = newContext[float32]()
        let a = ctx.variable(-2)
        let b = ctx.variable(-0.000654123)
        let c = ctx.variable(0.123456)
        let d = ctx.variable(10)

        let ra = relu(a)
        let rb = relu(b)
        let rc = relu(c)
        let rd = relu(d)

        check: ra.value == 0
        check: rb.value == 0
        check: rc.value == c.value
        check: rd.value == d.value
        check: ra.grad.wrt(a) == 0
        check: rb.grad.wrt(b) == 0
        check: rc.grad.wrt(c) == 1
        check: rd.grad.wrt(d) == 1