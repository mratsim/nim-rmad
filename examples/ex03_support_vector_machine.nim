## This is a port of Andrej Karpathy SVM (Support Vector Machine) example to my autograd framework.
## This is the complex, modular version!
## Original source: https://karpathy.github.io/neuralnets/
## It's very! rough, I'm figuring out the best API for those weights updates and stochastic gradient descent.

import ../rmad, random, math

proc forwardNeuron[T](x,y: T, a, b, c: Variable[T]): Variable[T] =
    ## forward pass
    let
        ax = a * x
        by = b * y
        axpby = ax + by
        axpbypc = axpby + c
    return axpbypc

proc backward[T](label: T, a, b, c, unit_out:Variable[T]): (T, T, T)=
    ## backpropagation pass - thanks to autodiff, only regularization has to be done

    # compute the pull (regularization) based on what the circuit output was
    var pull: T = 0
    if(label == 1 and unit_out.value < 1):
        pull = 1 # the score was too low: pull up
    if(label == -1 and unit_out.value > -1):
        pull = -1 # the score was too high for a positive example, pull down

    let g_out = unit_out.grad(pull) # Autograd at work, yeah!

    # add regularization pull for parameters: towards zero and proportional to value
    let reg_a = g_out.wrt(a) - a.value
    let reg_b = g_out.wrt(b) - b.value

    let gc = g_out.wrt(c)

    return (reg_a, reg_b, gc)

proc parameterUpdate[T](a,b,c:Variable[T]; ga,gb,gc: T): (T, T, T)=
    ## Update parameter with regularized gradients

    let step_size: T = 0.01
    let
        upd_a = a.value + step_size * ga
        upd_b = b.value + step_size * gb
        upd_c = c.value + step_size * gc

    return (upd_a, upd_b, upd_c)

proc learnFrom[T](data: seq[array[2,T]], labels: seq[T], a,b,c:Variable[T]): (Variable[T], Variable[T], Variable[T]) =
    ## Wraps everything in a neat learning package

    # Pick a random data point
    let 
        i = random(data.len)
        x = data[i][0]
        y = data[i][1]
        label = labels[i]
    
    let unit_out = forwardNeuron(x,y,a,b,c)
    let (reg_a, reg_b, gc) = backward(label, a,b,c,unit_out)
    let (new_a, new_b, new_c) = parameterUpdate(a, b, c, reg_a, reg_b, gc)

    let #Create a fresh context and wrap variables in it
        new_ctx = newContext[T]()
        v_a = new_ctx.variable(new_a)
        v_b = new_ctx.variable(new_b)
        v_c = new_ctx.variable(new_c)
    
    return (v_a, v_b, v_c)

var data = newSeq[array[2, float32]]()
var labels = newSeq[float32]()

data.add([1.2'f32, 0.7]); labels.add(1)
data.add([-0.3'f32, -0.5]); labels.add(-1)
data.add([3.0'f32, 0.1]); labels.add(1)
data.add([-0.1'f32, -1.0]); labels.add(-1)
data.add([-1.0'f32, 1.1]); labels.add(-1)
data.add([2.1'f32, -3]); labels.add(1)

proc evalTrainingAccuracy[T](data: seq[array[2,T]], labels: seq[T], a,b,c:Variable[T]): T =
    ## Compute the classification accuracy
    var num_correct: int = 0

    for i in 0..<data.len:
        let x = data[i][0]
        let y = data[i][1]
        let true_label = labels[i]

        # see if the prediction matches the provided label
        var predicted_label: float32
        if (forwardNeuron(x, y, a, b, c).value > 0):
            predicted_label = 1
        else:
            predicted_label = -1

        if predicted_label == true_label:
            inc num_correct
    
    return num_correct.T / data.len.T

## Initialize with random weights a, b, c
let ctx = newContext[float32]()
var a = ctx.variable(1)
var b = ctx.variable(-3)
var c = ctx.variable(-1)

for iter in 0..<925:
    ## Learning loop - aka Stochastic Gradient Descent
    (a, b, c) = learnFrom(data, labels, a, b, c)
    if iter mod 100 == 0:
        echo("\ntraining accuracy at iter ", $iter, ": ", $evalTrainingAccuracy(data, labels, a,b,c))
        echo("a = ", $a.value,", b = ", $b.value,", c = ", $c.value)

#### Output
# training accuracy at iter 0: 0.5
# a = 0.9900000095367432, b = -2.970000028610229, c = -1.0
# 
# training accuracy at iter 100: 0.8333333134651184
# a = 0.6099905371665955, b = -0.8328615427017212, c = -1.049999952316284
# 
# training accuracy at iter 200: 0.8333333134651184
# a = 0.661786675453186, b = -0.1738961637020111, c = -0.9700000286102295
# 
# training accuracy at iter 300: 0.8333333134651184
# a = 0.679929256439209, b = -0.1482911854982376, c = -0.820000171661377
# 
# training accuracy at iter 400: 0.8333333134651184
# a = 0.625005841255188, b = -0.03091072663664818, c = -0.8600001335144043
# 
# training accuracy at iter 500: 0.8333333134651184
# a = 0.5997063517570496, b = -0.04239700734615326, c = -0.8300001621246338
# 
# training accuracy at iter 600: 0.8333333134651184
# a = 0.6394171118736267, b = -0.152386263012886, c = -0.7600002288818359
# 
# training accuracy at iter 700: 0.8333333134651184
# a = 0.6148572564125061, b = -0.1321496069431305, c = -0.7700002193450928
# 
# training accuracy at iter 800: 0.8333333134651184
# a = 0.5911520719528198, b = -0.1409305483102798, c = -0.7800002098083496
# 
# training accuracy at iter 900: 1.0
# a = 0.6575953364372253, b = -0.04309666901826859, c = -0.6900002956390381