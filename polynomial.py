"""
polynomial.py — a from-scratch Polynomial class with a forgiving string parser.

Stores a polynomial as a dict {power: coefficient}.
Coefficients are kept as exact fractions.Fraction internally so that things like
1/3 stay 1/3 (important for exact integration). They are only turned into ints/
nice strings for display.

Highlights
----------
- Build from str ("x**3-2x**2-x+2"), list/tuple of coefs, dict, number, or Polynomial.
- Arithmetic: + - * ** , scalar / (true division), unary -, and reflected r-ops
  so 2 + p, 3 * p, (2,0,1,0) + p all work.
- p.derivative()           -> exact derivative
- p.integral()             -> indefinite integral, constant left symbolic as 'C'
- p.integral(5)            -> indefinite integral with the constant pinned to 5
- p.integral(a, b)         -> definite integral (a number): the area from a to b
- p.definite_integral(a,b) -> number (area)
- p.roots()                -> tries numpy if installed (handles everything, incl.
                              complex), else falls back to: quadratic formula for
                              degree<=2, Horner/rational-root peeling otherwise.
"""

from fractions import Fraction

# ----------------------------------------------------------------------------
# coefficient helpers
# ----------------------------------------------------------------------------
def _to_frac(x):
    """Turn a coefficient into an exact Fraction. Floats are made exact-ish via str."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        return Fraction(int(x))
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        # Fraction(str(0.1)) -> 1/10 rather than the ugly binary expansion
        return Fraction(str(x))
    return Fraction(x)


def _num(f):
    """Fraction -> int if whole, else Fraction (used internally for tidy values)."""
    f = _to_frac(f)
    return int(f) if f.denominator == 1 else f


def _fmt(f):
    """Format a coefficient for display: int if whole, else 'p/q'."""
    f = _to_frac(f)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


##EXPRESSION PARSER (recursive descent)
def _tokenize(s, var):
    """Break an expression string into a list of (kind, text) tokens. Recognises the variable (which may be multi-character, e.g. 'x1' or 'vel'), numbers, operators, parens, and known function/constant names."""
    s = s.replace('^', '**').replace(' ', '')
    tokens = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '*' and i + 1 < n and s[i + 1] == '*':
            tokens.append(('OP', '**')); i += 2; continue
        if c in '+-*/':
            tokens.append(('OP', c)); i += 1; continue
        if c == '(':
            tokens.append(('LP', '(')); i += 1; continue
        if c == ')':
            tokens.append(('RP', ')')); i += 1; continue
        # the variable can be multi-character (x1, y2, vel) -> match it greedily first
        if var and s.startswith(var, i):
            # make sure we're not matching a prefix of a longer identifier
            nxt = i + len(var)
            after = s[nxt] if nxt < n else ''
            if not (after.isalnum()):     # var not followed by more letters/digits
                tokens.append(('VAR', var)); i = nxt; continue
        if c.isdigit() or c == '.':
            j = i
            while j < n and (s[j].isdigit() or s[j] == '.'):
                j += 1
            # scientific notation: 1e3, 2.5e-4
            if j < n and s[j] in 'eE':
                j += 1
                if j < n and s[j] in '+-':
                    j += 1
                while j < n and s[j].isdigit():
                    j += 1
            tokens.append(('NUM', s[i:j])); i = j; continue
        # a name: letters then optional trailing digits (function/const, or stray identifier)
        if c.isalpha():
            j = i
            while j < n and s[j].isalpha():
                j += 1
            while j < n and s[j].isdigit():
                j += 1
            tokens.append(('NAME', s[i:j])); i = j; continue
        raise SyntaxError(f"unexpected character {c!r} in expression")
    return tokens


class _Parser:
    """Recursive-descent parser that turns a token list into a Polynomial, honouring precedence, parentheses, unary minus, and implicit multiplication."""
    def __init__(self, tokens, var, poly_cls):
        self.toks = tokens
        self.i = 0
        self.var = var
        self.P = poly_cls

    def peek(self):
        """Return the current token without consuming it (or (None, None) at the end)."""
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def next(self):
        """Consume and return the current token, advancing the position."""
        t = self.toks[self.i]; self.i += 1; return t

    def parse(self):
        """Parse the whole token stream into a Polynomial; error if tokens are left over."""
        node = self.expr()
        if self.i != len(self.toks):
            raise SyntaxError("unexpected trailing input in expression")
        return node

    def expr(self):
        """Grammar rule: handle + and - (lowest precedence)."""
        node = self.term()
        while self.peek()[1] in ('+', '-'):
            op = self.next()[1]
            rhs = self.term()
            node = node + rhs if op == '+' else node - rhs
        return node

    def _starts_atom(self):
        """True if the next token could begin a new factor (used to detect implicit multiplication like (x+1)x)."""
        kind = self.peek()[0]
        return kind in ('NUM', 'VAR', 'LP', 'NAME')

    def term(self):
        """Grammar rule: handle *, /, and implicit multiplication."""
        node = self.unary()
        while True:
            kind, val = self.peek()
            if val in ('*', '/'):
                self.next()
                rhs = self.unary()
                if val == '*':
                    node = node * rhs
                else:
                    node = self._divide(node, rhs)
            elif self._starts_atom():        # implicit multiplication: 2x, (x+1)x
                rhs = self.unary()
                node = node * rhs
            else:
                break
        return node

    def unary(self):
        """Grammar rule: handle a leading + or - sign."""
        kind, val = self.peek()
        if val == '+':
            self.next(); return self.unary()
        if val == '-':
            self.next(); return -self.unary()
        return self.power()

    def power(self):
        """Grammar rule: handle ** (exponent must be a constant non-negative integer)."""
        base = self.atom()
        if self.peek()[1] == '**':
            self.next()
            exp = self.unary()                # exponent
            e = self._as_int_const(exp)
            return base ** e
        return base

    def atom(self):
        """Grammar rule: a number, the variable, a named constant/function, or a parenthesised sub-expression."""
        kind, val = self.next()
        if kind == 'NUM':
            return self.P({0: _num_to_frac(val)}, self.var)
        if kind == 'VAR':
            return self.P({1: Fraction(1)}, self.var)
        if kind == 'NAME':
            from math import sin, cos, tan, log, pi, e, sqrt, exp  # noqa: F401
            funcs = {'sin': sin, 'cos': cos, 'tan': tan, 'log': log,
                     'sqrt': sqrt, 'exp': exp}
            consts = {'pi': pi, 'e': e}
            if val in consts:                       # bare constant: pi, e
                return self.P({0: Fraction(str(float(consts[val])))}, self.var)
            if val in funcs:                         # function call: sin(2), log(10)
                if self.peek()[1] != '(':
                    raise ValueError(f"'{val}' must be called like {val}(...)")
                self.next()                          # consume '('
                arg = self.expr()
                if self.peek()[1] != ')':
                    raise SyntaxError(f"missing ')' after {val}(")
                self.next()                          # consume ')'
                if max(arg.d.keys()) != 0:
                    raise ValueError(f"{val}(...) needs a constant argument")
                result = funcs[val](float(arg.d[0]))
                return self.P({0: Fraction(str(float(result)))}, self.var)
            raise ValueError(f"unknown name {val!r} (not a function or constant)")
        if kind == 'LP':
            node = self.expr()
            if self.peek()[1] != ')':
                raise SyntaxError("missing closing ')'")
            self.next()
            return node
        raise SyntaxError(f"unexpected token {val!r}")

    # ---- helpers --------------------------------------------------------
    def _divide(self, node, rhs):
        """Divide a polynomial by a constant only; reject division by anything containing the variable (no rational functions)."""
        # only allow division by a constant polynomial (degree 0)
        if max(rhs.d.keys()) != 0:
            raise ValueError("division by a non-constant (rational functions unsupported)")
        denom = rhs.d[0]
        if denom == 0:
            raise ZeroDivisionError("division by zero in expression")
        return self.P({p: c / denom for p, c in node.d.items()}, self.var)

    def _as_int_const(self, node):
        """Extract a non-negative integer exponent from a constant node, or raise if it isn't one."""
        if max(node.d.keys()) != 0:
            raise ValueError("exponent must be a constant, not contain the variable")
        v = node.d[0]
        if v.denominator != 1 or v < 0:
            raise ValueError("exponent must be a non-negative integer")
        return int(v)


def _num_to_frac(s):
    """Parse a numeric token string into an exact Fraction (handles ints, ratios, decimals, scientific notation)."""
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return Fraction(str(float(s)))


def parse_expression(s, var, poly_cls):
    """Parse string s into a poly_cls instance."""
    toks = _tokenize(s, var)
    if not toks:
        return poly_cls({0: Fraction(0)}, var)
    return _Parser(toks, var, poly_cls).parse()

##CONVERT
def str_to_d(expression, var, register):  # register False: A = a, True: A != a
    """
    Parse a polynomial expression string into a {power: coef} dict.

    Uses a recursive-descent parser, so it handles full arithmetic, nested
    parentheses, and implicit multiplication, e.g.:
        (3**4+4/2-.5)x**2-(x+1)x(x**2+1)
        3x(x+1) - (x-1)**2
    Division by a constant is allowed; division by something containing the
    variable raises ValueError (no rational functions).
    """
    if expression == '':
        return {0: Fraction(0)}
    if not register:
        expression = expression.lower()
    result = parse_expression(expression, var, Polynomial)
    return dict(result.d)


def _parse_coef(s):
    """Parse a coefficient string exactly. Falls back to eval for sin/cos/pi/etc."""
    s = s.strip()
    if s in ('', '+'):
        return Fraction(0)
    if s == '-':
        return Fraction(-1)
    try:
        return Fraction(s)            # handles '3', '-2', '7/4'
    except (ValueError, ZeroDivisionError):
        try:
            return Fraction(str(float(s)))   # handles '2.4', '1e3'
        except ValueError:
            from math import sin, cos, tan, log, pi, e  # noqa: F401
            return _to_frac(eval(s))         # last resort: '2*pi', 'log(10)'


def list_to_d(expression):  # list/tuple; max power = len - 1
    """Build a {power: coef} dict from a list/tuple of coefficients (highest power first) or a list of (power, coef) pairs."""
    if type(expression[0]) in (list, tuple):
        return {int(k): _to_frac(v) for k, v in dict(expression).items()}
    else:
        return {i - 1: _to_frac(expression[-i]) for i in range(len(expression), 0, -1)}


# names the parser understands as constants/functions, not as the variable
_KNOWN_NAMES = {'pi', 'e', 'sin', 'cos', 'tan', 'log', 'sqrt', 'exp', 'abs'}


def detect_var(expression):
    """
    Guess the variable from a polynomial string so you don't have to pass var=.
    e.g. detect_var('2t**2') -> 't', detect_var('x1**2-1') -> 'x1'.

    Rules:
      * one identifier (a letter-run with optional trailing digits, not a known
        constant/function) is taken as the variable
      * a string with no variable at all (e.g. '5', '2*pi') defaults to 'x'
      * two different candidate identifiers is ambiguous -> ValueError (this class
        is single-variable; for true multivariable polynomials use SymPy)

    Known names ignored: pi, e, sin, cos, tan, log, sqrt, exp, abs.
    A trailing-digit identifier like 'x1' is NOT confused with a known name.
    """
    import re
    s = expression.lower()
    # blank out scientific-notation exponents (1e3, 2.5e-4) so 'e3' isn't seen as a var
    s = re.sub(r'(?<=[0-9.])e[+-]?[0-9]+', ' ', s)
    # identifiers: letters then optional digits (x, vel, x1, y2)
    names = re.findall(r'[a-z]+[0-9]*', s)
    candidates = set()
    for name in names:
        base = name.rstrip('0123456789')
        if name in _KNOWN_NAMES:        # exact known constant/function
            continue
        if base in _KNOWN_NAMES and name == base:
            continue
        candidates.add(name)
    if not candidates:
        return 'x'
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ValueError(
        f"ambiguous variable: found {sorted(candidates)}; this class is "
        "single-variable, so pass var= explicitly (or use SymPy for multivariable)")


def to_d(expression, var=None, register=False):
    """Convert any supported input into a normalised {power: coef} dict. If var is None and the input is a string, the variable letter is autodetected."""
    if var is None:
        var = detect_var(expression) if isinstance(expression, str) else "x"
    try:
        if expression is None or expression is False or (
                not isinstance(expression, (list, tuple, dict, str, Polynomial))
                and not isinstance(expression, bool) and not expression
                and not isinstance(expression, (int, float, Fraction))):
            d = {0: Fraction(0)}
        elif isinstance(expression, bool):
            d = {0: Fraction(int(expression))}
        elif isinstance(expression, (int, float, Fraction)):
            d = {0: _to_frac(expression)}
        elif isinstance(expression, (tuple, list)):
            d = list_to_d(expression) if len(expression) else {0: Fraction(0)}
        elif isinstance(expression, str):
            d = str_to_d(expression, var, register)
        elif isinstance(expression, dict):
            d = {int(k): _to_frac(v) for k, v in expression.items()}
        elif isinstance(expression, Polynomial):
            d = dict(expression.d)
        else:
            d = {0: _to_frac(expression)}
    except (SyntaxError, ValueError, ZeroDivisionError):
        # parser errors are informative ("unclosed ')'", "rational functions
        # unsupported", "exponent must be a non-negative integer") -> surface them
        raise
    except Exception:
        tname = str(type(expression)).split("'")[1]
        raise TypeError(
            "can only generate a polynomial dictionary from NoneType, bool, int, "
            f'float, Fraction, str, dict, or Polynomial (not "{tname}")')
    return d_fill(d)


def d_to_str(d, var):  # self.__str__
    s = ""
    for power, coef in d.items():
        if coef == 0:
            continue
        if power == 0:
            curr = _fmt(coef)
        else:
            if coef == 1:
                curr = ""
            elif coef == -1:
                curr = "-"
            elif coef.denominator != 1:        # fractional coef -> wrap in parens
                curr = f"({_fmt(coef)})"        # so (4/3)x**3, not 4/3x**3
            else:
                curr = _fmt(coef)
            curr += var
            if power != 1:
                curr += f"**{power}"
        s += curr + "+"
    s = s.replace("+-", "-")
    if s == "":
        s = "0+"
    return s[:-1]


##OPERATIONS
def d_mul(d, expression, var='x'):
    """Multiply a {power: coef} dict by another polynomial-like value, returning a new dict."""
    expression = to_d(expression, var)
    new_d_list = []
    for power, coef in expression.items():
        new_d_list.append({p + power: c * coef for p, c in d.items()})
    from functools import reduce
    return reduce(add_dicts, new_d_list) if new_d_list else {0: Fraction(0)}


def d_pow(d, num):  # only positive integers
    if num == 0:
        return {0: Fraction(1)}
    if num != abs(int(num)):
        raise ValueError('Only positive integer values can be used')
    num = abs(int(num))
    new_d = dict(d)
    for _ in range(num - 1):
        new_d = d_mul(new_d, d)
    return new_d


def add_dicts(d1, d2):
    """Add two {power: coef} dicts term by term, returning a new normalised dict."""
    d1 = dict(d1)
    d2 = dict(d2)
    for key in list(d1.keys()):
        if key in d2:
            d1[key] = d1[key] + d2.pop(key)
    d1.update(d2)
    return d_fill(d1)


def put(d, value):
    """Evaluate the polynomial dict at a given value (Fraction-exact for real input, supports complex)."""
    value = _to_frac(value) if not isinstance(value, complex) else value
    answer = 0
    for power, coef in d.items():
        answer += coef * value ** power
    return _num(answer) if not isinstance(answer, complex) else answer


def d_fill(d):  # fills intermediate powers with 0, trims leading zeros
    d = {int(k): _to_frac(v) for k, v in d.items()}
    for i in range(int(max(d.keys())), -1, -1):
        if i not in d:
            d[i] = Fraction(0)
    d = dict(sorted(d.items(), reverse=True))
    for power, coef in tuple(d.items()):
        if coef == 0 and len(d) != 1:
            del d[power]
        else:
            break
    return d


##INSTRUMENTS
def divisors(x):
    """Return a tuple of the positive divisors of |x| (used by the rational-root search)."""
    x = abs(int(x))
    return tuple(i for i in range(1, x + 1) if x % i == 0)


def find_integer_root(d):  # rational root theorem, integer case
    try:
        const = d[0]
        if const.denominator != 1:
            return None
        divs = divisors(const)
    except Exception:
        return 0
    for num in divs:
        if put(d, num) == 0:
            return num
        if put(d, -num) == 0:
            return -num
    return None


##GORNER (synthetic division by integer roots)
def gorner_check(d):  # only positive integer powers supported
    for power in d.keys():
        if int(power) != power or power < 0:
            return False
    return True


def gorner_reduce(d):
    """Divide out one integer root using Horner / synthetic division, returning the reduced Polynomial (or None if no integer root)."""
    x = find_integer_root(d)
    if x is None:
        return None
    d = d_fill(d)
    if not gorner_check(d):
        raise ValueError('only positive integer-like powers can be used')
    new_d = {max(d.keys()) - 1: d[max(d.keys())]}
    for power in tuple(d.keys())[1:]:
        if power == 0:
            break
        new_d[power - 1] = d[power] + x * new_d[power]
    return Polynomial(new_d)


def gorner(d):
    """Repeatedly peel off integer roots, returning (remaining Polynomial, list of integer roots found)."""
    answers = []
    while True:
        x = find_integer_root(d)
        if x is None:
            return Polynomial(d), answers
        reduced = gorner_reduce(d)
        if reduced is None or d == reduced.d:
            return Polynomial(d), answers
        d = reduced.d
        answers.append(x)


def iter_round(it):
    """Tidy an iterable of Fractions into ints where whole, leaving everything else as-is."""
    return tuple(_num(e) if isinstance(e, Fraction) else e for e in it)


##CLASS MAIN BODY
class Polynomial():
    """A polynomial with exact (Fraction) coefficients, stored as {power: coef}. Supports arithmetic, calculus, evaluation, and root-finding, and is built from strings, lists, dicts, numbers, or other Polynomials."""
    ##CLASS BUILT-INS
    def __init__(self, expression=None, var=None, register=False):
        """Build a polynomial from a string, list/tuple, dict, number, bool, None, or another Polynomial. If var is None and a string is given, the variable letter is autodetected (e.g. Polynomial('2t**2') -> variable 't'). register=True makes parsing case-sensitive."""
        if var is None and isinstance(expression, str):
            var = detect_var(expression)
        self.d = to_d(expression, var, register)
        if isinstance(expression, Polynomial):
            self.var = expression.var
        else:
            self.var = var if var is not None else "x"

    def __repr__(self):
        """Unambiguous representation that can be passed back to eval()."""
        return f"Polynomial('{str(self)}', '{self.var}')"

    def __str__(self):
        """Human-readable polynomial string (what print shows)."""
        return d_to_str(self.d, self.var)

    def __len__(self):
        """Number of stored terms."""
        return len(self.d)

    def __eq__(self, other):
        """Equal if the other value represents the same polynomial."""
        try:
            return self.d == to_d(other, self.var)
        except Exception:
            return NotImplemented

    def __hash__(self):
        """Hash based on the term dict, so Polynomials can go in sets/dict keys."""
        return hash(tuple(sorted(self.d.items())))

    def __neg__(self):
        """Return the polynomial with every coefficient negated."""
        return Polynomial({p: -c for p, c in self.d.items()}, self.var)

    def __abs__(self):  # leading coef made positive (no longer mutates self)
        if self.d[max(self.d.keys())] > 0:
            return Polynomial(self.d, self.var)
        return -self

    def __bool__(self):
        """False only for the zero polynomial."""
        return self.d != {0: Fraction(0)}

    def __add__(self, expression):
        """Add another polynomial-like value (number, list, dict, str, Polynomial)."""
        return Polynomial(add_dicts(self.d, to_d(expression, self.var)), self.var)
    __radd__ = __add__

    def __sub__(self, expression):
        """Subtract another polynomial-like value."""
        return Polynomial(add_dicts(self.d, d_mul(to_d(expression, self.var), -1, self.var)), self.var)

    def __rsub__(self, expression):
        """Support number - Polynomial."""
        return (-self) + expression

    def __mul__(self, expression):
        """Multiply by another polynomial-like value."""
        return Polynomial(d_mul(self.d, expression, self.var), self.var)
    __rmul__ = __mul__

    def __truediv__(self, scalar):
        """Scalar (number) division only. Exact via Fraction."""
        try:
            s = _to_frac(scalar)
        except Exception:
            return NotImplemented
        if s == 0:
            raise ZeroDivisionError("polynomial division by zero")
        return Polynomial({p: c / s for p, c in self.d.items()}, self.var)

    def __pow__(self, power):
        """Raise to a non-negative integer power."""
        return Polynomial(d_pow(self.d, power), self.var)

    def __getitem__(self, power):
        """Return the coefficient of a given power (0 for an absent integer power)."""
        try:
            return _num(self.d[power])
        except KeyError:
            if isinstance(power, int):
                return 0
            raise

    def __iter__(self):
        """Iterate over coefficients (highest power first)."""
        return iter(self.coefs())

    def __call__(self, value):  # lets you do p(3) to evaluate
        return self.put(value)

    ##ITERABILITIES
    def powers(self):
        """Return a tuple of the powers present (highest first)."""
        return iter_round(self.d.keys())

    def coefs(self):
        """Return a tuple of the coefficients (highest power first)."""
        return iter_round(self.d.values())

    def non_zero(self):  # FIX: was self.pows()
        return {power: _num(c) for power, c in self.d.items() if c != 0}

    def pairs(self):
        """Return a tuple of (power, coefficient) pairs."""
        return tuple((p, _num(c)) for p, c in self.d.items())

    def items(self):
        """Return a tuple of term strings like '3x**2'."""
        return tuple(f"{self[power]}{self.var}**{power}" for power in self.powers())

    @property
    def degree(self):
        """The highest power present (the degree of the polynomial)."""
        return max(self.d.keys())

    ##IN-PLACE MANIPULATIONS
    def add(self, expression):
        """In-place: add another polynomial-like value to this one."""
        self.d = add_dicts(self.d, to_d(expression, self.var))

    def sub(self, expression):
        """In-place: subtract another polynomial-like value from this one."""
        self.add(d_mul(to_d(expression, self.var), -1))

    def mul(self, expression):
        """In-place: multiply this polynomial by another value."""
        self.d = d_mul(self.d, expression, self.var)

    def pow(self, power):
        """In-place: raise this polynomial to an integer power."""
        self.d = d_pow(self.d, power)

    def round(self, ndigits=None):
        """In-place: round every coefficient to ndigits decimal places."""
        self.d = {power: _to_frac(round(float(coef), ndigits)) for power, coef in self.d.items()}

    ##CALCULUS
    def derivative(self, order=1):
        """Exact derivative. Pass order=2 for the second derivative, etc."""
        d = self.d
        for _ in range(order):
            d = {p - 1: c * p for p, c in d.items() if p != 0}
            if not d:
                d = {0: Fraction(0)}
        return Polynomial(d, self.var)

    def integral(self, a='C', b=None):
        """
        Integrate the polynomial.

        Two modes, chosen by whether a second argument is given:

        INDEFINITE (one or no argument)  -> antiderivative
            f.integral()        -> symbolic, constant left as 'C' (pin later
                                   with .sfc(x, F) / .solve_for_constant / .at)
            f.integral(5)       -> antiderivative with the constant fixed to 5

        DEFINITE (two arguments = bounds) -> a number, the area F(b) - F(a)
            f.integral(1, 2)    -> value of the integral from x=1 to x=2

        Examples
        --------
        a = Polynomial('2.4*x**2')          # acceleration a_x(t)
        v = a.integral()                    # -> 0.8*t**3 + C   (symbolic C)
        v = a.integral(1.3)                 # pin constant = v0x
        area = Polynomial('x**2').integral(0, 3)   # 9
        """
        # DEFINITE: both bounds given -> return F(b) - F(a) as a number
        if b is not None:
            d = {p + 1: coef / (p + 1) for p, coef in self.d.items()}
            d[0] = d.get(0, Fraction(0))
            F = Polynomial(d, self.var)
            return F.put(b) - F.put(a)

        # INDEFINITE: 'a' plays the role of the integration constant
        c = a
        d = {p + 1: coef / (p + 1) for p, coef in self.d.items()}
        const = d.get(0, Fraction(0))
        if isinstance(c, str):
            # symbolic constant: return a SymbolicIntegral wrapper
            return SymbolicIntegral(Polynomial(d if d else {0: Fraction(0)}, self.var),
                                    const, c, self.var)
        d[0] = const + _to_frac(c)
        return Polynomial(d, self.var)

    def definite_integral(self, a, b):
        """Numeric value of the definite integral from a to b. Same as integral(a, b)."""
        return self.integral(a, b)

    # short aliases:  p.der(...) == p.derivative(...) ,  p.i(...) == p.integral(...)
    # note: 'd' is NOT used for derivative because self.d is the coefficient dict.
    der = derivative
    i = integral

    ##EVALUATION
    def put(self, value):
        """Evaluate the polynomial at a value. p(value) does the same thing."""
        return put(self.d, value)

    def table(self, start, stop=None, step=1, backward=False):
        """Return a dict of x: f(x). Boundaries included. Accepts floats."""
        if stop is None:
            start, stop = 0, start
        if start > stop:
            if step < 0:
                step = -step
            start, stop = stop, start
        table = {}
        curr = start
        while curr <= stop:
            table[curr] = self.put(curr)
            curr += step
        if backward:
            table = {v: k for k, v in table.items()}
        return table

    ##ROOTS
    def find_integer_root(self):
        """Find one integer root via the rational-root theorem, or None."""
        return find_integer_root(self.d)

    def gorner_check(self):
        """True if all powers are non-negative integers."""
        return gorner_check(self.d)

    def gorner_reduce(self):
        """Divide out one integer root by synthetic division (or None)."""
        return gorner_reduce(self.d)

    def gorner(self):
        """Peel off all integer roots; return (remaining Polynomial, roots found)."""
        return gorner(self.d)

    def roots(self, prefer_numpy=True):
        """
        Find all roots.

        Strategy:
          * degree 0  -> no roots (or 'all' if it's the zero polynomial)
          * degree 1  -> solve directly (exact)
          * degree 2  -> quadratic formula (exact when discriminant is a perfect
                         square, else float; complex if discriminant < 0)
          * degree >=3 -> try numpy.roots (handles real+complex robustly). If numpy
                         isn't installed, fall back to peeling integer roots with
                         gorner and report whatever's left unsolved.

        Returns a list. Exact roots come back as int/Fraction, the rest as
        float/complex.
        """
        if not gorner_check(self.d):
            raise ValueError("roots() needs non-negative integer powers")
        deg = self.degree

        if deg == 0:
            return [] if self.d[0] != 0 else ['all real numbers']

        if deg == 1:           # a*x + b = 0
            a, b = self.d[1], self.d.get(0, Fraction(0))
            return [_num(-b / a)]

        if deg == 2:           # a*x**2 + b*x + c
            a = self.d[2]
            b = self.d.get(1, Fraction(0))
            c = self.d.get(0, Fraction(0))
            disc = b * b - 4 * a * c
            if disc >= 0:
                r = _exact_sqrt(disc)
                if r is not None:          # perfect square -> exact roots
                    return sorted({_num((-b + r) / (2 * a)),
                                   _num((-b - r) / (2 * a))}, key=float)
                rt = float(disc) ** 0.5
                return sorted({(-float(b) + rt) / (2 * float(a)),
                               (-float(b) - rt) / (2 * float(a))})
            rt = (-float(disc)) ** 0.5
            re = -float(b) / (2 * float(a))
            im = rt / (2 * float(a))
            return [complex(re, im), complex(re, -im)]

        # degree >= 3
        if prefer_numpy:
            try:
                import numpy as np
                # numpy wants coefficients high->low power, as floats
                coeffs = [float(self.d.get(p, 0)) for p in range(deg, -1, -1)]
                raw = np.roots(coeffs)
                out = []
                for z in raw:
                    if abs(z.imag) < 1e-9:
                        val = round(z.real, 10)
                        out.append(int(val) if val == int(val) else float(val))
                    else:
                        out.append(complex(round(z.real, 10), round(z.imag, 10)))
                return out
            except ImportError:
                pass  # fall through to the built-in method

        # built-in fallback: peel off integer roots, then handle a final quadratic
        remaining, found = gorner(self.d)
        found = [int(x) for x in found]
        if remaining.degree <= 2 and remaining.degree >= 1:
            found += remaining.roots(prefer_numpy=False)
        elif remaining.degree >= 3:
            found.append(
                f"unsolved factor (install numpy for full roots): {remaining}")
        return found


class SymbolicIntegral:
    """
    Result of an indefinite integral with an unresolved constant 'C'.

    - print() shows it as  <polynomial> + C
    - .at(C_value) pins the constant and returns a real Polynomial
    - .put(x, C=...) evaluates directly
    - behaves like the polynomial-without-constant for further integration:
      .integral() integrates again (treating the unknown C as a coefficient on x).
    """
    def __init__(self, poly, const, symbol, var):
        self.poly = poly          # the integrated polynomial (its 0-power term is `const`)
        self.const = const        # the definite part of the constant term
        self.symbol = symbol      # e.g. 'C'
        self.var = var

    def at(self, c_value):
        """
        Pin the constant of integration to a known value and return a plain
        Polynomial.

        Example:
            f.integral().at(5)   # sets C = 5 outright
        """
        d = dict(self.poly.d)
        d[0] = self.const + _to_frac(c_value)
        return Polynomial(d, self.var)

    def solve_for_constant(self, x, F):
        """
        Solve for the unknown constant C using a known condition F(x) = F, then
        return the pinned Polynomial. This is the standard physics workflow.

        Parameters
        ----------
        x : the point at which the value of the antiderivative is known
        F : the known value of the antiderivative at x (F is the "mother
            function" value, e.g. velocity, position, charge — not necessarily v)

        Example
        -------
        a = Polynomial('9 - 1.6*x')                  # acceleration a(t)
        v = a.integral().solve_for_constant(0, 5)    # v(0) = 5  ->  C solved
        """
        base = self.poly.put(x)
        c_value = _to_frac(F) - _to_frac(base)   # C such that F(x) == F
        return self.at(c_value)

    # short alias:  f.integral().sfc(0, 5)  ==  .solve_for_constant(0, 5)
    sfc = solve_for_constant

    def put(self, x, C=0):
        """
        Evaluate the antiderivative at x for a given constant C (default 0),
        without pinning C permanently. Returns a number.
        """
        return self.poly.put(x) + _to_frac(C)

    def __str__(self):
        """Render as '<polynomial> + C'."""
        body = d_to_str(self.poly.d, self.var)
        if body == '0':
            return self.symbol
        return f"{body} + {self.symbol}"

    def __repr__(self):
        """Unambiguous representation that can be passed back to eval()."""
        return f"SymbolicIntegral({self.poly!r}, +{self.symbol})"


def _exact_sqrt(frac):
    """Return exact Fraction sqrt if frac is a perfect square, else None."""
    frac = _to_frac(frac)
    from math import isqrt
    n, d = frac.numerator, frac.denominator
    rn, rd = isqrt(n), isqrt(d)
    if rn * rn == n and rd * rd == d:
        return Fraction(rn, rd)
    return None


if __name__ == "__main__":
    p = Polynomial
    print((p('x**2+2') + (2, 0, 1, 0)) * 2 / 3 - 2)
    print(p('x**3-2x**2-x+2').roots())
