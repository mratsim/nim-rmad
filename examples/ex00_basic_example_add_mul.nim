## source: https://github.com/kkimdev/autograd

import ../rmad.nim

let ctx = newContext[float32]()

let x1 = ctx.variable(1.5)
let x2 = ctx.variable(2)

let five = ctx.variable(5)

let y = x1 * x2 + x1 + five

echo y.value # 9.5

let g = y.grad
echo g.wrt(x1) # 3
echo g.wrt(x2) # 1.5