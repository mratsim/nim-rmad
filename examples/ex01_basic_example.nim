import ../rmad.nim

let t = newContext[float32]()

let x = t.variable(0.5)
let y = t.variable(4.2);
let z = x * y + x.sin();

let gradient = z.grad()

echo z.value # 2.57942533493042
echo gradient.wrt(x) # 5.077582359313965
echo gradient.wrt(y) # 0.5