# kinpoly

Exact polynomial algebra and polynomial-component vectors for kinematics problems.

- `polynomial.py` — `Polynomial` with `fractions.Fraction` coefficients, a forgiving string
  parser (`"x**3-2x**2-x+2"`, `^`, implicit multiplication, multi-character variables),
  derivatives, indefinite/definite integrals and roots.
- `vector.py` — `Vector` whose components are `Polynomial`s in `t`: dot, cross, magnitude,
  component-wise derivative and integral, `.sfc(t0, value)` to pin integration constants.

```python
from vector import Vector
a = Vector('2.4*t**2', '9 - 1.6*t')      # acceleration
v = a.integral().sfc(0, (1.3, 5))        # v(0) = (1.3, 5)
r = v.integral().sfc(0, (0, 0))          # r(0) = origin
r.y.put(7)                               # y at t = 7
```

Flat modules, standard library only. No tests or license yet.
