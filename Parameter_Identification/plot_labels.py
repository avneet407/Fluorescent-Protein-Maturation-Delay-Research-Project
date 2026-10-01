"""Turn parameter names into matplotlib mathtext for figure labels.

Parameter names are plain strings throughout the code ("I0", "kb", "alpha",
"G3*I0", "kb (Set 1)", ...); `math_label` renders them as typeset symbols
(I_0, k_b, alpha, G_3 I_0, ...) for display only, leaving the names used as
DataFrame columns and dict keys unchanged.
"""

import re

_GREEK = {"alpha"}
_FACTOR = re.compile(r"([A-Za-z])(\w*)")


def _factor_to_math(factor):
    if factor in _GREEK:
        return "\\" + factor
    m = _FACTOR.fullmatch(factor)
    if m is None:
        return None
    symbol, subscript = m.groups()
    return f"{symbol}_{{{subscript}}}" if subscript else symbol


def math_label(name):
    """Return `name` as mathtext, e.g. "G3*I0" -> "$G_{3}\\,I_{0}$".

    A trailing " (...)" qualifier is kept as plain text ("kb (Set 1)" ->
    "$k_{b}$ (Set 1)"). Names that don't look like a symbol are returned
    unchanged.
    """
    symbol, sep, rest = name.partition(" (")
    factors = [_factor_to_math(f) for f in symbol.split("*")]
    if not symbol or None in factors:
        return name
    return "$" + r"\,".join(factors) + "$" + (sep + rest if sep else "")
