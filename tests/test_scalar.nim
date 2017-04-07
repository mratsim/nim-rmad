import ../rmad.nim, unittest

# Approximate Floating point comparison
# Considerations https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
# Implementation of a general solution is probably best on R
# https://github.com/PredictiveEcology/fpCompare
proc `=~`(x, y: float): bool = (abs(x-y) < 1e-8)

suite "Scalar input autodifferentiation":
    test "Context initialization":
        when compiles(newContext[float32]()):
            check: true
    test "Basic operations - Addition":
        let ctx = newContext[float32]()
        let a = ctx.variable(10)
        let b = ctx.variable(5)

        let k = a + b
        check: k.value =~ 15'f32
        let gk = k.grad
        check: gk.wrt(a) =~ 1'f32
        check: gk.wrt(b) =~ 1'f32
    test "Basic operations - Substraction":
        let ctx = newContext[float32]()
        let a = ctx.variable(10)
        let b = ctx.variable(5)

        let p = a - b
        check: p.value =~ 5'f32
        let gp = p.grad
        check: gp.wrt(a) =~ 1'f32
        check: gp.wrt(b) =~ -1'f32
    test "Basic operations - Multiplication":
        let ctx = newContext[float32]()
        let a = ctx.variable(10)
        let b = ctx.variable(5)

        let m = a * b
        check: m.value =~ 50'f32
        let gm = m.grad
        check: gm.wrt(a) =~ 5'f32
        check: gm.wrt(b) =~ 10'f32
    test "Basic operations - Division":
        let ctx = newContext[float32]()
        let a = ctx.variable(10)
        let b = ctx.variable(5)

        let n = a / b
        check: n.value =~ 2'f32
        let gn = n.grad

        check: gn.wrt(a) =~ 1 / 5
        check: gn.wrt(b) =~ 10 * -1/(5*5)

    test "Several gradients, based on same variables in same context":
        let ctx = newContext[float32]()
        let a = ctx.variable(10)
        let b = ctx.variable(5)

        let k = a + b
        let p = a - b
        let m = a * b
        let n = a / b

        check: k.value =~ 15'f32
        check: p.value =~ 5'f32
        check: m.value =~ 50'f32
        check: n.value =~ 2'f32

        let gk = k.grad
        let gp = p.grad
        let gm = m.grad
        let gn = n.grad

        check: gk.wrt(a) =~ 1'f32
        check: gk.wrt(b) =~ 1'f32
        check: gp.wrt(a) =~ 1'f32
        check: gp.wrt(b) =~ -1'f32
        check: gm.wrt(a) =~ 5'f32
        check: gm.wrt(b) =~ 10'f32
        check: gn.wrt(a) =~ 1 / 5
        check: gn.wrt(b) =~ 10 * -1/(5*5)

    test "Mixed variables and literals":
        let ctx = newContext[float32]()

        let x1 = ctx.variable(1.5)
        let x2 = ctx.variable(2)

        let y = x1 * x2 + x1 + 5

        check: y.value == 9.5

        let g = y.grad

        check: g.wrt(x1) =~ 3
        check: g.wrt(x2) =~ 1.5


#   TODO: test "Different contexts prevention":
#   TODO: gradient of a + a, a - a, a / a, a * a, a ^ 0
