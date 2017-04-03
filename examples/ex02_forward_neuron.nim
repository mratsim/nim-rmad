## This is a port of Andrej Karpathy forward neuron example to my autograd framework
## Original source: https://karpathy.github.io/neuralnets/

import ../rmad

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