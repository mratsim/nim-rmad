import ../rmad.nim, unittest, math

# Approximate Floating point comparison
# Considerations https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
# Implementation of a general solution is probably best on R
# https://github.com/PredictiveEcology/fpCompare
proc `=~`[T:SomeReal, U:SomeNumber](x, y: T|U): bool = (abs(x.T-y.T) < 1e-8)

suite "Scalar input autodifferentiation":
    test "Approximate comparison sanity check":
        check: not (1'f32 =~ 1.0000001)
        check: 1'f32 =~ 1.00000001

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

    test "Multiple gradients, based on same variables in same context":
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

    test "Mixed variables and literals + Automatic conversion":
        #TODO: support for integer literal
        let ctx = newContext[float32]()

        let a = ctx.variable(1.5)

        when compiles(2 * a):
            check: true

        let m = 2 * a
        let n = a * 2
        let p = 2 + a
        let q = a + 2
        let r = 2 - a
        let s = a - 2
        let t = a / 2
        let u = 2 / a

        check: m.grad.wrt(a) =~ 2
        check: n.grad.wrt(a) =~ 2
        check: p.grad.wrt(a) =~ 1
        check: q.grad.wrt(a) =~ 1
        check: r.grad.wrt(a) =~ -1
        check: s.grad.wrt(a) =~ 1
        check: t.grad.wrt(a) =~ 1/2
        check: u.grad.wrt(a) =~ -2 / (1.5 * 1.5)

    test "Gradients of a+a, a-a, a/a, a*a, a^0 ...":
        let ctx = newContext[float32]()

        let a = ctx.variable(10)

        let m = a+a+a+a+a
        let n = 5 * a
        let p = a*a*a*a*a
        let q = pow(a,5)
        let r = a - a
        let s = a / a
        let t = pow(a,0)

        let gm = m.grad
        let gn = n.grad
        let gp = p.grad
        let gq = q.grad
        let gr = r.grad
        let gs = s.grad
        let gt = t.grad


        check: gm.wrt(a) =~ 5
        check: gm.wrt(a) =~ gn.wrt(a)
        check: gp.wrt(a) =~ 50000
        check: gp.wrt(a) =~ gq.wrt(a)
        check: gr.wrt(a) =~ 0
        check: gs.wrt(a) =~ 0
        check: gt.wrt(a) =~ 0

    test "Trigonometric functions":
        #TODO: support for integer literal
        let ctx = newContext[float32]()

        let a = ctx.variable(PI / 6)

        let m = cos(a)
        let n = sin(a)
        let p = tan(a)
        let q = arccos(a)
        let r = arcsin(a)
        let s = arctan(a)
        let t = arccos(cos(a))
        let u = cos(arccos(a))
        let v = cos(a).pow(2'f32) + sin(a).pow(2'f32)
        let w = sin(a) / cos(a)

        check: m.grad.wrt(a) =~ -0.5 # -sin(pi/6)
        check: n.grad.wrt(a) =~ sqrt(3'f32)/2 # cos(pi/6)
        check: p.grad.wrt(a) =~ 4/3 # = 1/cos(pi/6)^2 = 1 + tan(pi/6)^2
        check: q.grad.wrt(a) =~ -1 / sqrt( 1 - (PI/6).pow(2'f32)) # -1 / sqrt(1 - x^2)
        check: r.grad.wrt(a) =~ 1 / sqrt( 1 - (PI/6).pow(2'f32)) # 1 / sqrt(1 - x^2)
        check: s.grad.wrt(a) =~ 1 / ( 1 + (PI/6).pow(2'f32)) # 1 / s(1 + x^2)
        check: t.grad.wrt(a) =~ 1
        # check: u.grad.wrt(a) =~ 1 # Precision 1e-6 required 
        check: v.grad.wrt(a) =~ 0
        # check: w.grad.wrt(a) =~ 1 + tan(PI/6).pow(2) # Precision 1e-6 required 

    test "Exponential and logarithm":
        let ctx = newContext[float32]()

        let a = ctx.variable(10.3)
        let b = ctx.variable(0.5)

        let m = exp(a)
        let n = ln(a)
        let p = a.pow(b)
        let q = log10(a)
        let r = sqrt(a)

        check: m.grad.wrt(a) =~ m.value
        check: n.grad.wrt(a) =~ 1 / 10.3
        check: p.grad.wrt(a) =~ 0.5 * 10.3.pow(-0.5)
        check: q.grad.wrt(a) =~ 1/ (10.3 * ln(10'f32))
        check: r.grad.wrt(a) =~ 1/(2 * sqrt(10.3))

        check: p.grad.wrt(a) =~ r.grad.wrt(a)

    test "Hyperbolic functions":
        let ctx = newContext[float32]()

        let a = ctx.variable(10)

        let m = a.cosh
        let n = a.sinh
        let p = a.tanh


        check: m.grad.wrt(a) =~ a.value.sinh
        check: n.grad.wrt(a) =~ a.value.cosh
        check: p.grad.wrt(a) =~ 1 / a.value.cosh.pow(2) # 1/cosh^2 x = 1 - tanh^2 x # Needs 1e-8 precision

#   TODO: test "Divide by 0, ln(0)"
#   TODO: test "Different contexts prevention":

