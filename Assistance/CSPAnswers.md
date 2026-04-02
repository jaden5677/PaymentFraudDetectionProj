# Linear Programming Solutions

## Problem 1: Primo Insurance Company

**Decision Variables:**
- x1 = units of special risk insurance
- x2 = units of mortgages

**Formulation:**
- Maximize: 5x1 + 2x2
- Subject to:
  - 3x1 + 2x2 ≤ 2400 (Underwriting)
  - x2 ≤ 800 (Administration)
  - 2x1 ≤ 1200 (Claims)
  - x1, x2 ≥ 0

```python
import pulp as p

prob = p.LpProblem("Primo_Insurance", p.LpMaximize)

x = p.LpVariable.dict("x", [1, 2], lowBound=0, cat=p.LpContinuous)

prob += 5*x[1] + 2*x[2]

prob += 3*x[1] + 2*x[2] <= 2400
prob += x[2] <= 800
prob += 2*x[1] <= 1200

print(prob)
status = prob.solve()
print(p.LpStatus[status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Optimal profit =", p.value(prob.objective))
```

---

## Problem 2: Weenies and Buns (Group 2)

**Decision Variables:**
- x1 = hot dogs per week
- x2 = hot dog buns per week

**Formulation:**
- Maximize: 0.80x1 + 0.30x2
- Subject to:
  - 0.1x2 ≤ 200 (Flour: 200 lbs/week, 0.1 lb per bun)
  - 0.25x1 ≤ 800 (Pork: 800 lbs/week, 1/4 lb per hot dog)
  - 3x1 + 2x2 ≤ 12000 (Labor: 5 employees × 40 hrs × 60 min = 12000 min/week)
  - x1, x2 ≥ 0

```python
import pulp as p

prob = p.LpProblem("Weenies_and_Buns", p.LpMaximize)

x = p.LpVariable.dict("x", [1, 2], lowBound=0, cat=p.LpContinuous)

prob += 0.80*x[1] + 0.30*x[2]

prob += 0.1*x[2] <= 200
prob += 0.25*x[1] <= 800
prob += 3*x[1] + 2*x[2] <= 12000

print(prob)
status = prob.solve()
print(p.LpStatus[status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Optimal profit = $", p.value(prob.objective))
```

---

## Problem 3: Omega Manufacturing (Group 3)

**Decision Variables:**
- x1 = units of product 1
- x2 = units of product 2
- x3 = units of product 3

**Formulation:**
- Maximize: 50x1 + 20x2 + 25x3
- Subject to:
  - 9x1 + 3x2 + 5x3 ≤ 500 (Milling machine)
  - 5x1 + 4x2 ≤ 350 (Lathe)
  - 3x1 + 2x3 ≤ 150 (Grinder)
  - x3 ≤ 20 (Sales potential for product 3)
  - x1, x2, x3 ≥ 0

```python
import pulp as p

prob = p.LpProblem("Omega_Manufacturing", p.LpMaximize)

x = p.LpVariable.dict("x", [1, 2, 3], lowBound=0, cat=p.LpContinuous)

prob += 50*x[1] + 20*x[2] + 25*x[3]

prob += 9*x[1] + 3*x[2] + 5*x[3] <= 500
prob += 5*x[1] + 4*x[2]            <= 350
prob += 3*x[1]           + 2*x[3]  <= 150
prob += x[3] <= 20

print(prob)
status = prob.solve()
print(p.LpStatus[status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Optimal profit = $", p.value(prob.objective))
```

---

## Problem 4: Joyce and Marvin Daycare (Group 4)

**Decision Variables:**
- x1 = slices of bread
- x2 = tbsp peanut butter
- x3 = tbsp strawberry jelly
- x4 = graham crackers
- x5 = cups of milk
- x6 = cups of juice

**Formulation:**
- Minimize: 5x1 + 4x2 + 7x3 + 8x4 + 15x5 + 35x6 (cost in cents)
- Subject to:
  - 70x1 + 100x2 + 50x3 + 60x4 + 150x5 + 100x6 ≥ 400 (Min calories)
  - 70x1 + 100x2 + 50x3 + 60x4 + 150x5 + 100x6 ≤ 600 (Max calories)
  - 10x1 + 75x2 + 20x4 + 70x5 ≤ 0.3(70x1 + 100x2 + 50x3 + 60x4 + 150x5 + 100x6) (Fat ≤ 30%)
  - 3x3 + 2x5 + 120x6 ≥ 60 (Vitamin C ≥ 60mg)
  - 3x1 + 4x2 + x4 + 8x5 + x6 ≥ 12 (Protein ≥ 12g)
  - x1 = 2 (Exactly 2 slices of bread)
  - x2 ≥ 2x3 (At least twice as much PB as jelly)
  - x5 + x6 ≥ 1 (At least 1 cup of liquid)
  - All variables ≥ 0

```python
import pulp as p

prob = p.LpProblem("Daycare_Lunch", p.LpMinimize)

x = p.LpVariable.dict("x", [1, 2, 3, 4, 5, 6], lowBound=0, cat=p.LpContinuous)

prob += 5*x[1] + 4*x[2] + 7*x[3] + 8*x[4] + 15*x[5] + 35*x[6]

total_cal = 70*x[1] + 100*x[2] + 50*x[3] + 60*x[4] + 150*x[5] + 100*x[6]
fat_cal = 10*x[1] + 75*x[2] + 0*x[3] + 20*x[4] + 70*x[5] + 0*x[6]

prob += total_cal >= 400
prob += total_cal <= 600
prob += fat_cal <= 0.30 * total_cal
prob += 0*x[1] + 0*x[2] + 3*x[3] + 0*x[4] + 2*x[5] + 120*x[6] >= 60
prob += 3*x[1] + 4*x[2] + 0*x[3] + 1*x[4] + 8*x[5] + 1*x[6] >= 12
prob += x[1] == 2
prob += x[2] >= 2*x[3]
prob += x[5] + x[6] >= 1

print(prob)
status = prob.solve()
print(p.LpStatus[status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Minimum cost (cents) =", p.value(prob.objective))
```


