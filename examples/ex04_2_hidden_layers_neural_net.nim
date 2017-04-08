## This is a port of Andrej Karpathy SVM (2 layer Neural Network) example to my autograd framework.
## Original source: https://karpathy.github.io/neuralnets/
## Non-modular version

import ../rmad, random, math

## Dataset
var data = newSeq[array[2, float32]]()
var labels = newSeq[float32]()

data.add([1.2'f32, 0.7]); labels.add(1)
data.add([-0.3'f32, -0.5]); labels.add(-1)
data.add([3.0'f32, 0.1]); labels.add(1)
data.add([-0.1'f32, -1.0]); labels.add(-1)
data.add([-1.0'f32, 1.1]); labels.add(-1)
data.add([2.1'f32, -3]); labels.add(1)


## Weights Initialization
let ctx = newContext[float32]()
var
    a1 = ctx.variable(random(1'f32) - 0.5)
    b1 = ctx.variable(random(1'f32) - 0.5)
    c1 = ctx.variable(random(1'f32) - 0.5)

    a2 = ctx.variable(random(1'f32) - 0.5)
    b2 = ctx.variable(random(1'f32) - 0.5)
    c2 = ctx.variable(random(1'f32) - 0.5)

    a3 = ctx.variable(random(1'f32) - 0.5)
    b3 = ctx.variable(random(1'f32) - 0.5)
    c3 = ctx.variable(random(1'f32) - 0.5)

    a4 = ctx.variable(random(1'f32) - 0.5)
    b4 = ctx.variable(random(1'f32) - 0.5)
    c4 = ctx.variable(random(1'f32) - 0.5)
    d4 = ctx.variable(random(1'f32) - 0.5)

type Weight_list[T] = (Variable[T],Variable[T],Variable[T],Variable[T],Variable[T],Variable[T],Variable[T],Variable[T],Variable[T],Variable[T],Variable[T],Variable[T],Variable[T])

proc forwardNeuralNet[T](x,y: T, weight_watcher: Weight_list[T]) : Variable[T] =
    let (a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, d4) = weight_watcher
    let
       n1 = relu(a1 * x + b1 * y + c1) # Activation of first hidden neuron
       n2 = relu(a2 * x + b2 * y + c2) # 2nd neuron
       n3 = relu(a3 * x + b3 * y + c3) # 3nd neuron
       score = a4 * n1 + b4 * n2 + c4 * n3 + d4 # score
    return score

## Evaluation routine
proc evalTrainingAccuracy[T](data: seq[array[2,T]], labels: seq[T], weight_watcher: Weight_list[T]): T =
    ## Compute the classification accuracy
    var num_correct: int = 0

    for i in 0..<data.len:
        let x = data[i][0]
        let y = data[i][1]
        let true_label = labels[i]

        # see if the prediction matches the provided label
        var predicted_label: float32
        if (forwardNeuralNet(x, y, weight_watcher).value > 0):
            predicted_label = 1
        else:
            predicted_label = -1

        if predicted_label == true_label:
            inc num_correct
    
    return num_correct.T / data.len.T

## Learning loop
for iter in 0..<825:
    let 
        i = random(data.len)
        x = data[i][0]
        y = data[i][1]
        label = labels[i]

    let weights = (a1, b1, c1, a2, b2, c2, a3, b3, c3, a1, b4, c4, d4)
    let score = forwardNeuralNet(x,y,weights)
    
    var pull: float32 = 0
    if (label == 1 and score.value < 1):
        pull = 1
    if (label == -1 and score.value > -1):
        pull = -1
    
    # Backprop - 40 lines of Andrej replaced by 1 - Hell yeah !
    let g = score.grad(pull)

    # Parameters update
    let step_size = 0.01

    let 
        da1 = a1.value + step_size * (g.wrt(a1) - a1.value)
        db1 = b1.value + step_size * (g.wrt(b1) - b1.value)
        dc1 = c1.value + step_size * (g.wrt(c1) - c1.value)

        da2 = a2.value + step_size * (g.wrt(a2) - a2.value)
        db2 = b2.value + step_size * (g.wrt(b2) - b2.value)
        dc2 = c2.value + step_size * (g.wrt(c2) - c2.value)

        da3 = a3.value + step_size * (g.wrt(a3) - a3.value)
        db3 = b3.value + step_size * (g.wrt(b3) - b3.value)
        dc3 = c3.value + step_size * (g.wrt(c3) - c3.value)

        da4 = a4.value + step_size * (g.wrt(a4) - a4.value)
        db4 = b4.value + step_size * (g.wrt(b4) - b4.value)
        dc4 = c4.value + step_size * (g.wrt(c4) - c4.value)
        dd4 = d4.value + step_size * (g.wrt(d4))

    let # Create a fesh context and wrap variables in it
        new_ctx = newContext[float32]()

    a1 = new_ctx.variable(da1)
    b1 = new_ctx.variable(db1)
    c1 = new_ctx.variable(dc1)

    a2 = new_ctx.variable(da2)
    b2 = new_ctx.variable(db2)
    c2 = new_ctx.variable(dc2)

    a3 = new_ctx.variable(da3)
    b3 = new_ctx.variable(db3)
    c3 = new_ctx.variable(dc3)

    a4 = new_ctx.variable(da4)
    b4 = new_ctx.variable(db4)
    c4 = new_ctx.variable(dc4)
    d4 = new_ctx.variable(dd4)

    let new_weights = (a1, b1, c1, a2, b2, c2, a3, b3, c3, a1, b4, c4, d4)

    if iter mod 50 == 0:
        echo("\ntraining accuracy at iter ", $iter, ": ", $evalTrainingAccuracy(data, labels, new_weights))
        echo("a1 = ", $a1.value,", b1 = ", $b1.value,", c1 = ", $c1.value)
        echo("a2 = ", $a2.value,", b2 = ", $b2.value,", c2 = ", $c2.value)
        echo("a3 = ", $a3.value,", b3 = ", $b3.value,", c3 = ", $c3.value)
        echo("a4 = ", $a4.value,", b4 = ", $b4.value,", c4 = ", $c4.value,", d4 = ", $d4.value,"\n")

## Output
# training accuracy at iter 0: 0.5
# a1 = -0.2646308243274689, b1 = 0.3196182548999786, c1 = -0.1649954319000244
# a2 = 0.486814022064209, b2 = -0.05441995710134506, c2 = -0.4180980920791626
# a3 = 0.09271980077028275, b3 = -0.3014206290245056, c3 = 0.02563019469380379
# a4 = 0.3211687505245209, b4 = 0.1837961077690125, c4 = -0.009962900541722775, d4 = -0.444730281829834
# 
# 
# training accuracy at iter 50: 0.5
# a1 = -0.1832568794488907, b1 = 0.2027656883001328, c1 = -0.09128256887197495
# a2 = 0.4080002903938293, b2 = -0.06478568911552429, c2 = -0.1970494985580444
# a3 = 0.06737832725048065, b3 = -0.1848164796829224, c3 = 0.01531886402517557
# a4 = 0.1943090558052063, b4 = 0.2648850381374359, c4 = 0.05296581983566284, d4 = -0.324730396270752
# 
# 
# training accuracy at iter 100: 0.5
# a1 = -0.1340912878513336, b1 = 0.1320429444313049, c1 = -0.04670972749590874
# a2 = 0.3396918177604675, b2 = -0.09783696383237839, c2 = -0.06970030069351196
# a3 = 0.05682585760951042, b3 = -0.1219291761517525, c3 = 0.008706693537533283
# a4 = 0.117558166384697, b4 = 0.2803196310997009, c4 = 0.07174245268106461, d4 = -0.3447303771972656
# 
# 
# training accuracy at iter 150: 0.5
# a1 = -0.0955626517534256, b1 = 0.08568311482667923, c1 = -0.02299026027321815
# a2 = 0.3144910931587219, b2 = -0.05292530730366707, c2 = -0.01127622555941343
# a3 = 0.0604662150144577, b3 = -0.06841083616018295, c3 = 0.003814512630924582
# a4 = 0.07112341374158859, b4 = 0.2878564298152924, c4 = 0.05993730202317238, d4 = -0.4047303199768066
# 
# 
# training accuracy at iter 200: 0.5
# a1 = -0.07817653566598892, b1 = 0.05997328832745552, c1 = -0.006514265667647123
# a2 = 0.2981939911842346, b2 = -0.06700637191534042, c2 = 0.02230074256658554
# a3 = 0.06160338595509529, b3 = -0.04661565274000168, c3 = 0.004150445573031902
# a4 = 0.04303008317947388, b4 = 0.2940790355205536, c4 = 0.06519178301095963, d4 = -0.4647302627563477
# 
# 
# training accuracy at iter 250: 0.5
# a1 = -0.06583240628242493, b1 = 0.0436549037694931, c1 = 0.00275947293266654
# a2 = 0.2869641780853271, b2 = -0.05920331180095673, c2 = 0.057417381554842
# a3 = 0.06203274428844452, b3 = -0.02997418120503426, c3 = 0.007613655179738998
# a4 = 0.02603345736861229, b4 = 0.2933030724525452, c4 = 0.06445816904306412, d4 = -0.4847302436828613
# 
# 
# training accuracy at iter 300: 0.5
# a1 = -0.04442095011472702, b1 = 0.02629773505032063, c1 = 0.005909543950110674
# a2 = 0.3215005695819855, b2 = -0.0758029967546463, c2 = 0.05643314868211746
# a3 = 0.07061243057250977, b3 = -0.02609966695308685, c3 = 0.007738025393337011
# a4 = 0.01575039885938168, b4 = 0.3321461379528046, c4 = 0.07386208325624466, d4 = -0.444730281829834
# 
# 
# training accuracy at iter 350: 0.5
# a1 = -0.03623022139072418, b1 = 0.01649450324475765, c1 = 0.01041063852608204
# a2 = 0.3110205233097076, b2 = -0.04880065843462944, c2 = 0.05823444202542305
# a3 = 0.06832724064588547, b3 = -0.01651801541447639, c3 = 0.01020598690956831
# a4 = 0.009529086761176586, b4 = 0.3182055950164795, c4 = 0.07015471905469894, d4 = -0.5247302055358887
# 
# 
# training accuracy at iter 400: 0.5
# a1 = -0.02914893999695778, b1 = 0.009398338384926319, c1 = 0.01200282666832209
# a2 = 0.3056211769580841, b2 = -0.04823694750666618, c2 = 0.06402616202831268
# a3 = 0.06726063042879105, b3 = -0.01410865038633347, c3 = 0.01252258289605379
# a4 = 0.005765156354755163, b4 = 0.3143930435180664, c4 = 0.06935238838195801, d4 = -0.5847301483154297
# 
# 
# training accuracy at iter 450: 0.5
# a1 = -0.02637503109872341, b1 = 0.004754137713462114, c1 = 0.01370705384761095
# a2 = 0.2790005505084991, b2 = -0.01239083241671324, c2 = 0.06584294140338898
# a3 = 0.06154812127351761, b3 = -0.00458513991907239, c3 = 0.01303382683545351
# a4 = 0.003487954149022698, b4 = 0.2857750356197357, c4 = 0.06275614351034164, d4 = -0.684730052947998
# 
# 
# training accuracy at iter 500: 0.5
# a1 = -0.02586957067251205, b1 = 0.003479496575891972, c1 = 0.01421244814991951
# a2 = 0.2695895731449127, b2 = -0.03478614240884781, c2 = 0.06082858145236969
# a3 = 0.05931856855750084, b3 = -0.008982368744909763, c3 = 0.01292378641664982
# a4 = 0.002110233763232827, b4 = 0.2772274315357208, c4 = 0.0610608272254467, d4 = -0.7447299957275391
# 
# 
# training accuracy at iter 550: 0.5
# a1 = -0.02048605866730213, b1 = 0.001748061738908291, c1 = 0.01183106005191803
# a2 = 0.3549469709396362, b2 = -0.06329696625471115, c2 = 0.09400712698698044
# a3 = 0.0781635046005249, b3 = -0.01474439352750778, c3 = 0.02043034695088863
# a4 = 0.001276704017072916, b4 = 0.3713131248950958, c4 = 0.08183641731739044, d4 = -0.6047301292419434
# 
# 
# training accuracy at iter 600: 0.8333333134651184
# a1 = -0.01771620847284794, b1 = 0.001022275653667748, c1 = 0.01044379733502865
# a2 = 0.4096724689006805, b2 = -0.09269148856401443, c2 = 0.1044067442417145
# a3 = 0.09027532488107681, b3 = -0.02091290429234505, c3 = 0.0228415559977293
# a4 = 0.0007724136230535805, b4 = 0.4311497211456299, c4 = 0.09507398307323456, d4 = -0.5447301864624023
# 
# 
# training accuracy at iter 650: 0.8333333134651184
# a1 = -0.01469437312334776, b1 = -0.000285586720565334, c1 = 0.009196702390909195
# a2 = 0.4692673981189728, b2 = -0.1251134425401688, c2 = 0.1044383347034454
# a3 = 0.1034330353140831, b3 = -0.02787249535322189, c3 = 0.02291865833103657
# a4 = 0.0004673149378504604, b4 = 0.4945414364337921, c4 = 0.1090586557984352, d4 = -0.504730224609375
# 
# 
# training accuracy at iter 700: 0.8333333134651184
# a1 = -0.013951919041574, b1 = -0.0004027653194498271, c1 = 0.008712960407137871
# a2 = 0.4791225790977478, b2 = -0.08398861438035965, c2 = 0.08517917990684509
# a3 = 0.1056246906518936, b3 = -0.01869173906743526, c3 = 0.01871581561863422
# a4 = 0.0002827283751685172, b4 = 0.4916097521781921, c4 = 0.1083974316716194, d4 = -0.564730167388916
# 
# 
# training accuracy at iter 750: 0.8333333134651184
# a1 = -0.01383278891444206, b1 = 0.0003857366100419313, c1 = 0.008197114802896976
# a2 = 0.4974794089794159, b2 = -0.1223074421286583, c2 = 0.105856254696846
# a3 = 0.1096811592578888, b3 = -0.02707274071872234, c3 = 0.02330127544701099
# a4 = 0.0001710523793008178, b4 = 0.5208226442337036, c4 = 0.1148458272218704, d4 = -0.6047301292419434
# 
# 
# training accuracy at iter 800: 1.0
# a1 = -0.01101768109947443, b1 = -0.0001501557417213917, c1 = 0.0067451405338943
# a2 = 0.617705225944519, b2 = -0.1316249221563339, c2 = 0.1579932421445847
# a3 = 0.1361964046955109, b3 = -0.02908591367304325, c3 = 0.03481331467628479
# a4 = 0.0001034877059282735, b4 = 0.6486729979515076, c4 = 0.1430323123931885, d4 = -0.4847302436828613
