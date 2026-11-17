"""
code_scan.py
============
One canonical token scanner, with the known-null **built into the API**.

WHY THIS MODULE EXISTS
----------------------
Three separate scans in this project have fired on their own documentation:

  1. ``audit_lambda0_r6``'s ramp scan, on ``run_bnm_r6.py``'s docstring -- which names every
     ramp token precisely because it documents the guard against them;
  2. ``test_bnm_round6.T11``'s cache-key scan, where ``"val"`` matched "value" and
     "evaluate";
  3. ``test_bnm_round6.T19``'s evaluation scan, on ``stats_only_row``'s docstring -- which
     says "TAKES NO SCORER, SO IT CANNOT EVALUATE".

Each was fixed where it fired. That is a treadmill: the next scan someone writes starts
from raw text again, because raw text is what ``in`` operates on. So the fix belongs in a
shared helper, and the helper is shaped so the two properties that make a scan trustworthy
are not optional:

  * it scans CODE, not prose -- comments and string literals are tokenised away;
  * it requires a KNOWN-NULL -- :func:`scan` takes ``known_null_src`` as a REQUIRED
    keyword argument and raises if that source does not match. A scan that has never been
    seen to fire is not known to work, and with this API you cannot write one that has not.

Use :func:`scan` for new scans. :func:`code_only` is exposed for callers that need the
tokenised text itself.
"""
from __future__ import annotations

import io
import tokenize


class KnownNullDidNotFire(AssertionError):
    """The positive control matched nothing, so the scan proves nothing."""


def code_only(src: str) -> str:
    """The source's NAME, OP and NUMBER tokens, with comments and strings removed.

    Falls back to the raw text if the source does not tokenise -- conservative in the
    right direction, since anything present in code is still found.
    """
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER):
                out.append(t.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src
    return " ".join(out)


def code_token_set(src: str) -> set:
    """The distinct NAME/OP/NUMBER tokens of ``src``, lowercased.

    Exact membership, so ``"val"`` cannot match "value" -- the T11 failure.
    """
    return {t.lower() for t in code_only(src).split()}


def dedent_block(seg: str) -> str:
    """Left-align a function/class source segment so it tokenises as a module."""
    lines = seg.splitlines()
    if not lines:
        return seg
    pad = min((len(l) - len(l.lstrip()) for l in lines[1:] if l.strip()), default=0)
    return "\n".join([lines[0]] + [l[pad:] if len(l) > pad else l for l in lines[1:]])


def scan(src: str, tokens, *, known_null_src: str, exact: bool = True) -> dict:
    """Scan ``src`` for ``tokens`` in CODE only, with a mandatory positive control.

    Parameters
    ----------
    src : str
        The source that should be CLEAN.
    tokens : iterable of str
        Tokens to look for, matched case-insensitively.
    known_null_src : str
        Source that MUST contain at least one of ``tokens``. This is not optional: if it
        matches nothing, :class:`KnownNullDidNotFire` is raised rather than a clean result
        returned. A scan whose control does not fire is not evidence about ``src``.
    exact : bool
        ``True`` (default) matches whole tokens, so ``"val"`` does not match "value".
        ``False`` falls back to substring matching over the tokenised text, for tokens
        that are genuinely fragments (e.g. dotted attribute chains).

    Returns
    -------
    dict with ``hits`` (what ``src`` matched -- empty means clean), ``known_null_hits``,
    and ``clean``.
    """
    toks = [str(t).lower() for t in tokens]

    def find(text):
        if exact:
            have = code_token_set(text)
            return [t for t in toks if t in have]
        blob = code_only(text).lower()
        return [t for t in toks if t in blob]

    null_hits = find(known_null_src)
    if not null_hits:
        raise KnownNullDidNotFire(
            "the known-null matched none of %r, so this scan proves nothing about the "
            "source it just declared clean. Give it a control that must fail." % (toks,))

    hits = find(src)
    return {"hits": hits, "known_null_hits": null_hits, "clean": not hits,
            "exact": exact, "n_tokens": len(toks)}
