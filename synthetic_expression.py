"""User-defined gene expression signal u(t) -> ground-truth maturation states.

Lets the Synthetic Gene Expression tab generate ground-truth I/(X)/M
trajectories and a noisy fluorescence trace from an arbitrary, user-typed
u(t) formula, by driving Maturation_Models' 1-step/2-step ODEs with u as a
time-varying callable instead of a fixed constant (both models start from
I = X = M = B = 0, since nothing has been translated yet at t=0). The
resulting noisy trace is what the Kalman Filter tab filters -- using its
own, potentially different, calibrated rate constants -- so its plots can
compare the filter's reconstruction against this tab's ground truth.

u(t) expressions are restricted to a small whitelist of names (`t`, `pi`,
`e`, and basic math functions) and AST node types, with no builtins exposed
to `eval`, since the formula is free-form user input.
"""

import ast

import numpy as np

from Maturation_Models import simulate_1step, simulate_2step

EXPRESSION_HELP = (
    "A formula in `t` (time), e.g. `5 + 3*sin(2*pi*t/300*2)`. Allowed: `t`, "
    "`pi`, `e`, the functions `sin`, `cos`, `tan`, `exp`, `log`, `log10`, "
    "`sqrt`, `abs`, `sign`, `min`, `max`, and `+ - * / ** %` with "
    "parentheses. No other names or function calls are allowed."
)

_ALLOWED_NAMES = {
    "pi": np.pi, "e": np.e,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "min": np.minimum, "max": np.maximum,
}

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def _validate_expr(expr):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid u(t) expression: {e}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed syntax in u(t) expression: {type(node).__name__}.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError("u(t) expression calls an unknown/disallowed function.")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES and node.id != "t":
            raise ValueError(f"Unknown name '{node.id}' in u(t) expression. {EXPRESSION_HELP}")

    return tree


def compile_u_expression(expr):
    """Validates and compiles a u(t) formula string into a callable `u(t)`.

    `t` may be a Python float (queried by the ODE solver during integration)
    or a numpy array (for plotting/grid evaluation); constant expressions
    (e.g. "5") broadcast to match `t`'s shape either way.
    """
    tree = _validate_expr(expr)
    code = compile(tree, "<u_expr>", "eval")

    def u_func(t):
        namespace = dict(_ALLOWED_NAMES)
        namespace["t"] = t
        value = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 (AST-validated above)
        return np.asarray(value, dtype=float) + np.zeros_like(np.asarray(t, dtype=float))

    return u_func


def simulate_true_1step(u_expr, km, kb, kd, alpha, dt, n_steps):
    """Integrates the 1-step model driven by a user u(t) formula.

    Returns a dict: t, true_u, true_I, true_M, true_F (= alpha * true_M).
    """
    u_func = compile_u_expression(u_expr)
    t = np.arange(int(n_steps)) * dt
    params = {"u": u_func, "km": km, "kb": kb, "kd": kd, "alpha": alpha}
    _, I, M, B, F = simulate_1step(t, params, I0=0.0, M0=0.0, B0=0.0)
    return {"t": t, "true_u": u_func(t), "true_I": I, "true_M": M, "true_F": F}


def simulate_true_2step(u_expr, km1, km2, kb, kd, alpha, dt, n_steps):
    """Integrates the 2-step model driven by a user u(t) formula.

    Returns a dict: t, true_u, true_I, true_X, true_M, true_F (= alpha * true_M).
    """
    u_func = compile_u_expression(u_expr)
    t = np.arange(int(n_steps)) * dt
    params = {"u": u_func, "k1": km1, "k2": km2, "kb": kb, "kd": kd, "alpha": alpha}
    _, I, X, M, B, F = simulate_2step(t, params, I0=0.0, X0=0.0, M0=0.0, B0=0.0)
    return {"t": t, "true_u": u_func(t), "true_I": I, "true_X": X, "true_M": M, "true_F": F}
