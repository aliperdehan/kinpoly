"""
vector.py — an n-dimensional Vector whose components are Polynomials (in t),
built on top of polynomial.Polynomial.

Designed for 2D/3D kinematics: represent a(t), v(t), r(t) as vectors of
polynomials, then differentiate / integrate component-wise.

A component may be any Polynomial-compatible value (number, string, Polynomial),
so you can write:
    a = Vector('2.4*t**2', '9 - 1.6*t')      # acceleration (x, y)
    v = a.integral().sfc(0, (1.3, 5))         # integrate, pin v(0) = (1.3, 5)
    r = v.integral().sfc(0, (0, 0))           # integrate, pin r(0) = origin
    r.y.put(7)                                # vertical position at t = 7

Vectors of plain numbers also work (constant-acceleration problems):
    vi = Vector(2, 6); vf = Vector(-2, 2)
    a  = (vf - vi) / 13                        # constant acceleration
"""

from polynomial import Polynomial, detect_var


def _as_poly(x, var='t'):
    """Coerce a component to a Polynomial (so numbers, strings, Polynomials all work)."""
    return x if isinstance(x, Polynomial) else Polynomial(x, var)


def _detect_var_from(components):
    """
    Pick the variable for a Vector from its components: detect it from any string
    components, inherit it from any Polynomial components, and fall back to 't'
    (the kinematics default) when there's nothing to go on. Raises if two string
    components disagree on the variable.
    """
    found = set()
    for c in components:
        if isinstance(c, Polynomial):
            # only count it if the component actually uses its variable
            if max(c.d.keys()) > 0:
                found.add(c.var)
        elif isinstance(c, str):
            try:
                v = detect_var(c)
            except ValueError:
                continue  # ambiguous/constant string: skip, decide from others
            if any(ch.isalpha() for ch in c):
                found.add(v)
    if len(found) > 1:
        raise ValueError(
            f"components use different variables {sorted(found)}; pass var= explicitly")
    return found.pop() if found else 't'


class Vector:
    """
    A vector of Polynomial components. Supports +, -, scalar * and /, dot, cross
    (3D), magnitude, and component-wise derivative/integral for kinematics.
    """

    def __init__(self, *components, var=None):
        """
        Build from components: Vector(ax, ay) or Vector(ax, ay, az).
        Each component may be a number, a string like '9 - 1.6*t', or a Polynomial.
        A single list/tuple is also accepted: Vector([ax, ay, az]).

        The variable is autodetected from any string components (so Vector('x**2')
        works), falling back to 't' when there's nothing to detect — kinematics is
        the common case. Pass var= to force it.
        """
        if len(components) == 1 and isinstance(components[0], (list, tuple)):
            components = tuple(components[0])
        if var is None:
            var = _detect_var_from(components)
        self.var = var
        self.comps = [_as_poly(c, var) for c in components]
        if not self.comps:
            raise ValueError("a Vector needs at least one component")

    # ---- access -----------------------------------------------------------
    @property
    def x(self):
        """First component."""
        return self.comps[0]

    @property
    def y(self):
        """Second component (KeyError-safe: raises IndexError if absent)."""
        return self.comps[1]

    @property
    def z(self):
        """Third component."""
        return self.comps[2]

    def __getitem__(self, i):
        """Return the i-th component (0=x, 1=y, 2=z)."""
        return self.comps[i]

    def __len__(self):
        """Number of components (the dimension of the vector)."""
        return len(self.comps)

    def __iter__(self):
        """Iterate over the component Polynomials."""
        return iter(self.comps)

    # ---- arithmetic -------------------------------------------------------
    def _align(self, other):
        """
        Return (self_comps, other_comps) padded with zeros to the same length, so
        a 2D and a 3D vector can be combined: (i,j) + (i2,j2,k) -> (i+i2, j+j2, k).
        """
        if not isinstance(other, Vector):
            raise TypeError(
                "can only combine a Vector with another Vector; "
                f"got {type(other).__name__}")
        n = max(len(self), len(other))
        zero = Polynomial(0, self.var)
        a = list(self.comps) + [zero] * (n - len(self))
        b = list(other.comps) + [zero] * (n - len(other))
        return a, b

    def __add__(self, other):
        """Component-wise addition; shorter vector is zero-padded to match."""
        a, b = self._align(other)
        return Vector(*[x + y for x, y in zip(a, b)], var=self.var)

    def __sub__(self, other):
        """Component-wise subtraction; shorter vector is zero-padded to match."""
        a, b = self._align(other)
        return Vector(*[x - y for x, y in zip(a, b)], var=self.var)

    def __neg__(self):
        """Return the vector with every component negated."""
        return Vector(*[-a for a in self.comps], var=self.var)

    def __mul__(self, scalar):
        """Multiply every component by a scalar (number or Polynomial). Rejects Vector*Vector (use .dot or .cross)."""
        if isinstance(scalar, Vector):
            raise TypeError(
                "Vector * Vector is ambiguous; use .dot(other) or .cross(other)")
        return Vector(*[a * scalar for a in self.comps], var=self.var)
    __rmul__ = __mul__

    def __truediv__(self, scalar):
        """Divide every component by a scalar number."""
        return Vector(*[a / scalar for a in self.comps], var=self.var)

    # ---- vector products --------------------------------------------------
    def dot(self, other):
        """Dot product (zero-padded if dimensions differ): returns a Polynomial (or number if constant)."""
        a, b = self._align(other)
        total = Polynomial(0, self.var)
        for x, y in zip(a, b):
            total = total + x * y
        return total

    def cross(self, other):
        """3D cross product, returns a Vector. Both vectors must be 3D."""
        if len(self) != 3 or len(other) != 3:
            raise ValueError("cross product needs two 3D vectors")
        a1, a2, a3 = self.comps
        b1, b2, b3 = other.comps
        return Vector(a2 * b3 - a3 * b2,
                      a3 * b1 - a1 * b3,
                      a1 * b2 - a2 * b1, var=self.var)

    def mag2(self, t=None):
        """
        Squared magnitude. With no argument returns it as a Polynomial in t;
        with a value of t returns the number |v(t)|**2.
        """
        s = self.dot(self)
        return s.put(t) if t is not None else s

    def is_constant(self):
        """True if no component depends on the variable (the vector is a plain constant)."""
        return all(max(c.d.keys()) == 0 for c in self.comps)

    def mag(self, t=None):
        """
        Magnitude |vector|, as a number.

        For a constant vector you can omit t (its magnitude is the same
        everywhere). If any component depends on the variable, t is required so
        the magnitude is well-defined.
        """
        if t is None:
            if not self.is_constant():
                raise TypeError(
                    "this vector depends on the variable, so mag needs a value, "
                    "e.g. v.mag(4)")
            t = 0
        return float(self.mag2(t)) ** 0.5

    # ---- evaluation -------------------------------------------------------
    def put(self, t):
        """Evaluate every component at t, returning a tuple of numbers."""
        return tuple(c.put(t) for c in self.comps)

    def __call__(self, t):
        """Evaluate every component at t; same as .put(t)."""
        return self.put(t)

    # ---- calculus (component-wise) ---------------------------------------
    def derivative(self, order=1):
        """Differentiate each component (e.g. r -> v -> a)."""
        return Vector(*[c.derivative(order) for c in self.comps], var=self.var)
    der = derivative

    def integral(self, *args):
        """
        Integrate each component.

        - integral()        -> a SymbolicVector (each component has its own 'C',
                               pin them all at once with .sfc(t, F_vector))
        - integral(a, b)     -> definite integral: a tuple of numbers per component
        """
        if len(args) == 2:                       # definite -> numbers
            a, b = args
            return tuple(c.integral(a, b) for c in self.comps)
        return SymbolicVector([c.integral() for c in self.comps], self.var)
    i = integral

    # ---- display ----------------------------------------------------------
    def __str__(self):
        """Render in unit-vector notation, e.g. '(2.4t**2)i + (9-1.6t)j'."""
        unit = ['i', 'j', 'k'] + [f'e{n}' for n in range(4, len(self) + 1)]
        parts = []
        for c, u in zip(self.comps, unit):
            parts.append(f"({c}){u}")
        return " + ".join(parts)

    def __repr__(self):
        """Unambiguous representation listing the component strings."""
        return f"Vector({', '.join(repr(str(c)) for c in self.comps)})"


class SymbolicVector:
    """
    Result of integrating a Vector with no bounds: a vector of SymbolicIntegrals,
    each carrying its own unknown constant. Pin them all with one known vector
    condition via .sfc(t, F_vector).
    """

    def __init__(self, sym_components, var):
        """Wrap a list of SymbolicIntegral components (one per axis), each with its own unknown constant."""
        self.syms = sym_components
        self.var = var

    def solve_for_constant(self, t, F):
        """
        Pin every component's constant from a known vector value F at parameter t.
        F may be a tuple/list of numbers (e.g. v(0) = (1.3, 5)) or a Vector, in
        which case it is evaluated at t. Returns a plain Vector.
        """
        if isinstance(F, Vector):
            F = F.put(t)
        if len(F) != len(self.syms):
            raise ValueError(f"condition has {len(F)} components, vector has {len(self.syms)}")
        pinned = [sym.solve_for_constant(t, fval) for sym, fval in zip(self.syms, F)]
        return Vector(*pinned, var=self.var)
    sfc = solve_for_constant

    def at(self, constants):
        """Pin each component's constant directly from a tuple/list of values."""
        return Vector(*[sym.at(c) for sym, c in zip(self.syms, constants)], var=self.var)

    def __str__(self):
        """Render with each component's symbolic '+ C', e.g. '((1/3)t**3 + C)i + ...'."""
        unit = ['i', 'j', 'k']
        return " + ".join(f"({s}){u}" for s, u in zip(self.syms, unit))

    def __repr__(self):
        """Unambiguous representation listing the symbolic component strings."""
        return f"SymbolicVector({[str(s) for s in self.syms]})"
