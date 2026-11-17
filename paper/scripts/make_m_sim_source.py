r"""
make_m_sim_source.py -- the simulated SOURCE galaxies, as macros read from the generator.
=========================================================================================

    python paper/scripts/make_m_sim_source.py

Parses ``scripts/generate_arc_on_real.py`` with ``ast`` (it is never imported -- it needs
GalSim, which is not installed here) and asserts, before emitting:

  * the catalogue call is ``galsim.COSMOSCatalog(sample=...)`` with NO other keyword, i.e. no
    exclusion argument is passed and the class defaults apply;
  * every ``makeGalaxy`` call passes ``gal_type="parametric"`` -- the sources are fitted
    models, not real HST images;
  * the half-light-radius window is the ``min_hlr``/``max_hlr`` defaults of
    ``draw_source_hires``;
  * arc brightness is set by ``ARC_SNR_RANGE``, and the stored ``target_snr`` of all 30,000
    simulated lenses lies inside it (the data agree with the constant).

Writes ``paper/tables/M_sim_source.tex``.
"""
from __future__ import annotations

import ast
import os

import numpy as np

import _common as C

GEN = os.path.join(C.REPO_SCRIPTS, "generate_arc_on_real.py")


def _num(x: float) -> str:
    return f"{x:g}"


def main() -> int:
    tree = ast.parse(open(GEN, encoding="utf-8").read())
    catalogs, makes, snr, hlr = [], [], None, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "COSMOSCatalog":
                catalogs.append({k.arg: ast.literal_eval(k.value) for k in node.keywords})
            if node.func.attr == "makeGalaxy":
                makes.append({k.arg: ast.literal_eval(k.value) for k in node.keywords
                              if k.arg == "gal_type"})
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ARC_SNR_RANGE"
                                                for t in node.targets):
            snr = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "draw_source_hires":
            args = node.args.args[-len(node.args.defaults):]
            hlr = {a.arg: ast.literal_eval(d) for a, d in zip(args, node.args.defaults)}

    assert len(catalogs) == 1 and set(catalogs[0]) == {"sample"}, catalogs
    sample = catalogs[0]["sample"]
    assert makes and all(m.get("gal_type") == "parametric" for m in makes), makes
    lo_h, hi_h = float(hlr["min_hlr"]), float(hlr["max_hlr"])
    lo_s, hi_s = (float(v) for v in snr)

    stored = np.concatenate([s["target_snr"] for _n, s in C.sim_shards(["target_snr"])])
    assert stored.size == 30000, stored.size
    assert lo_s <= stored.min() and stored.max() <= hi_s, (stored.min(), stored.max())

    macros = "\n".join([
        rf"\newcommand{{\simSourceSample}}{{{sample}}}",
        rf"\newcommand{{\simSourceGalType}}{{parametric}}",
        rf"\newcommand{{\simSourceHlrMin}}{{{_num(lo_h)}}}",
        rf"\newcommand{{\simSourceHlrMax}}{{{_num(hi_h)}}}",
        rf"\newcommand{{\simArcSnrMin}}{{{_num(lo_s)}}}",
        rf"\newcommand{{\simArcSnrMax}}{{{_num(hi_s)}}}",
    ])
    C.write_table(os.path.join(C.TABLES, "M_sim_source.tex"), macros,
                  header_comment="Macros for Section 3's source description, parsed from "
                                 "scripts/generate_arc_on_real.py; sources in "
                                 "paper/scripts/make_m_sim_source.py.", size="")
    print(f"  COSMOSCatalog(sample={sample!r}) only; makeGalaxy gal_type=parametric; "
          f"HLR {lo_h}-{hi_h} arcsec; arc S/N {lo_s}-{hi_s} "
          f"(stored target_snr {stored.min():.3f}-{stored.max():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
