import pulp as p

prob = p.LpProblem("Lab1-eg1", p.LpMaximize)

x = p.LpVariable.dict("x", [1, 2], lowBound=0, cat = p.LpContinuous)

prob += 3000*x[1] + 5000*x[2]

prob += 3*x[1] + 2*x[2] <= 18
prob += x[1] <= 4
prob += x[2] <= 6

print(prob)

status = prob.solve()
print(p.LpStatus[status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print(p.value(prob.objective))
