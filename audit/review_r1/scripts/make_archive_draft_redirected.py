"""make_archive_draft_redirected.py -- run paper/scripts/make_archive.py --draft unchanged,
with its output path pointed outside the repository.

    python -B audit/review_r1/scripts/make_archive_draft_redirected.py <out.zip>

``make_archive.main`` reads the module global ``OUT_DRAFT`` at call time; it is set to
<out.zip> so the existing <repo>/paper_DRAFT.zip is not rewritten (rule zero of the
review-round-1 brief). Gate, packing and self-verification are the script's own code.
"""
import os
import sys

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "paper", "scripts"))
import make_archive as m  # noqa: E402

m.OUT_DRAFT = os.path.abspath(sys.argv[1])
sys.argv = [sys.argv[0], "--draft"]
raise SystemExit(m.main())
