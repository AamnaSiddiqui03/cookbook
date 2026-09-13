"""Build confidence_calibration.ipynb. Run once; the notebook itself is what's committed.

Kept as a script rather than hand-edited JSON so the notebook's structure stays reviewable
as a diff, the same reasoning as the rest of this recipe.
"""

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

OUT = "confidence_calibration.ipynb"

cells = []


def md(text):
    cells.append(new_markdown_cell(text.strip()))


def code(text):
    cells.append(new_code_cell(text.strip()))


md(
    """
# Confidence Calibration: Does the Threshold Hold?

Unsiloed's own blog, [*Confidence Score Reliability: The Missing Metric*](https://www.unsiloed.ai/blog/confidence-score-reliability-the-missing-metric-in-document-extraction),
tells buyers to ask every extraction vendor three questions:

1. Show me your confidence-versus-true-accuracy curve.
2. At what threshold do I hit sub-1% error on auto-accepted fields?
3. How stable is this across document variability?

And the cookbook itself ships four different answers to a question it never quite asks out loud —
what confidence counts as "trust this value":

| Where | Threshold |
|---|---|
| `skills/unsiloed/SKILL.md` | `score >= 0.85` |
| `skills/unsiloed-hitl-extract/SKILL.md` | `0.97`, called "deliberately stricter" |
| `kyc-app/app.py` | `LOW_CONF = 0.85` |
| `skills/unsiloed-hitl-extract/SKILL.md` | gate on `min(grounding_score, extraction_score)` |

None of the four is derived from data. The HITL skill even names the risk itself, at line 162:
if scores bunch in a narrow band around a threshold, picking a cutoff inside that band
"selects an arbitrary slice of near-identical scores, not the risky fields."

This notebook answers the three questions, measured rather than guessed. It generates a
corpus with known-correct values, degrades it from a clean PDF down to a phone photo, and
watches what happens to the confidence scores as the document gets harder to read.
"""
)

md(
    """
## 1. Setup

Everything here reads cached results by default, so the notebook runs top to bottom with no
API key and no wait. Cells that would call Unsiloed's API are marked, and skipped unless you
delete the cache.
"""
)

code(
    """
%pip install pandas numpy matplotlib pillow pymupdf reportlab python-dotenv requests -q
"""
)

code(
    """
import json
import pathlib

import matplotlib.pyplot as plt
from PIL import Image

ROOT = pathlib.Path(".")
RESULTS = ROOT / "results"
DOCS = ROOT / "documents"
SAMPLES = ROOT / "samples"

print("Setup complete.")
"""
)

md(
    """
## 2. The Problem With Measuring Calibration

To know whether a confidence score is honest, you need the *right* answer for every field, to
compare against. Hand-labelling documents is slow, error-prone, and the results usually
cannot be published.

So the corpus is generated rather than collected. `corpus/generate.py` writes the values
first — an invoice number, a due date, a line-item table — and lays the document out second.
The ground truth is exact by construction, because we wrote it ourselves, and the whole
corpus is reproducible from a fixed seed.

Twelve documents: five invoices, four bank statements, three ID cards. Enough field variety
(money, dates, IDs, free text, table rows) that a finding has to hold across types, not just
happen to be true of one.
"""
)

code(
    """
truth = json.loads((DOCS / "invoice_01.truth.json").read_text())
print(json.dumps(truth, indent=2)[:600])
"""
)

md(
    """
## 3. The Difficulty Ladder

A fixed corpus cannot show how a threshold *should* move as quality drops, because every
document in it differs in content as well as condition. So the same generated document is
put through four tiers, holding content constant and varying only how hard it is to read:

| Tier | What happens | Simulates |
|---|---|---|
| **T0** | Nothing — the original PDF | downloaded from a portal |
| **T1** | Rasterised at 150dpi, text layer gone | printed and scanned |
| **T2** | + skew, sensor noise, soft blur | a tired office scanner |
| **T3** | + perspective tilt, glare, a hand shadow, heavy JPEG | photographed on a phone |

`samples/degradation_ladder.png` shows one invoice through all four. Look at the right-hand
edge of T3: the glare has wiped out the entire unit-price column, while the descriptions
next to it are still readable. That is deliberate — a tier that is uniformly unreadable
would test nothing, and a tier that damages nothing would show no curve.
"""
)

code(
    """
ladder = Image.open(SAMPLES / "degradation_ladder.png")
ladder
"""
)

md(
    """
## 4. The Real Half: CORD Receipts

The generated ladder isolates *how quality affects scores*, but it cannot say whether
Unsiloed's scores are honest on documents from the wild — every generated document is, in
the end, a document we designed to be extractable.

For that, this notebook also scores 92 real receipts from [CORD](https://github.com/clovaai/cord),
a benchmark of photographed Indonesian receipts released by NAVER CLOVA AI Research under
CC-BY-4.0, with ground truth published by the researchers who built it. Three examples are
committed under `samples/cord/`; the rest are downloaded by `corpus/cord.py`.
"""
)

code(
    """
sample_receipt = SAMPLES / "cord" / "receipt_002.jpg"
truth = json.loads((SAMPLES / "cord" / "receipt_002.truth.json").read_text())
print(json.dumps(truth, indent=2, ensure_ascii=False))
Image.open(sample_receipt)
"""
)

md(
    """
This receipt is photographed in poor light with a hand's shadow falling across the totals —
the SUB TOTAL line is barely legible even to a person. That is exactly the kind of document
a reliability measurement needs: one that can actually defeat an extractor, some of the time.
"""
)

md(
    """
## 5. Running the Extraction (skipped if cached)

140 extraction calls total: 12 documents × 4 tiers, plus 92 CORD receipts. Every response is
cached under `results/extractions/`, so this cell does nothing if the cache already exists —
which it does in this repo, so the notebook reproduces every figure below with zero API spend.

To run it fresh: delete `results/extractions/` and run `python -m calibration.run` from a
terminal (takes a few minutes, needs `UNSILOED_API_KEY` in `.env`).
"""
)

code(
    """
cached = list((RESULTS / "extractions").glob("*.json")) if (RESULTS / "extractions").exists() else []
print(f"{len(cached)} extraction responses cached. Re-run calibration.run to refresh.")
"""
)

md(
    """
## 6. Scoring: What Counts as Correct

Every extracted field is compared to ground truth with a normalisation rule specific to its
type — spelled out in `calibration/compare.py` rather than hidden in analysis code, because a
calibration result is only as honest as its definition of "correct":

- **money** — currency symbols and thousands separators stripped, compared as a decimal
- **date** — parsed to a date object, so `21/07/2026` and `2026-07-21` match
- **id** — spaces, dashes, and case folded
- **text** — case and whitespace folded

A comparison that stayed strict about formatting too would count a display choice Unsiloed
never claimed to preserve as an extraction failure — which would measure the wrong thing.
"""
)

code(
    """
import pandas as pd
import sys
sys.path.insert(0, "..")

from calibration.analyse import load_rows  # noqa: E402

frame = load_rows()
print(f"{len(frame)} scored fields across {frame['document'].nunique()} documents")
frame.groupby("tier")["correct"].agg(accuracy="mean", n="size")
"""
)

md(
    """
## 7. Finding 1 — Which Score Actually Predicts Correctness?

Unsiloed's own HITL skill gates on `min(grounding_score, extraction_score)`. Is that better
than using either score alone? Measured as separation: given one correct field and one
incorrect field, how often does the correct one score higher?
"""
)

code(
    """
Image.open(RESULTS / "figures" / "01_score_discrimination.png")
"""
)

md(
    """
**All three are equivalent**, within noise. `grounding_score` and `extraction_score` answer
different questions — was the value located, was it read correctly once located — but
neither is more trustworthy than the other for the single decision that matters
operationally: should this value be trusted without a human looking at it?
"""
)

md(
    """
## 8. Finding 2 — Are the Scores Calibrated?

A confidence score is calibrated if 0.9 confidence really does mean "90% of the time, this
is correct." The reliability diagram plots mean confidence in a bucket against how often
fields in that bucket were actually correct, one line per tier. A perfectly calibrated score
sits on the diagonal.
"""
)

code(
    """
Image.open(RESULTS / "figures" / "02_reliability_grounding.png")
"""
)

md(
    """
**T0 and T1 hug the diagonal.** On clean and scanned documents, the score means what it
claims to mean. **T2 drifts, and T3 is badly miscalibrated** — a 0.4 confidence on a
photographed document does not carry the same meaning as a 0.4 on a clean PDF. The score is
still informative (higher generally means more likely correct — see Finding 1) but the
*number itself* stops being trustworthy as a probability once the document degrades enough.
"""
)

md(
    """
## 9. Finding 3 — What Threshold Actually Gets Sub-1% Error?

For every possible threshold, what fraction of fields does it auto-accept, and what's the
error rate among what it accepts?
"""
)

code(
    """
Image.open(RESULTS / "figures" / "03_threshold_sweep.png")
"""
)

code(
    """
pd.read_csv(RESULTS / "sub_1pct_thresholds.csv")
"""
)

md(
    """
On **T0 and T1**, a threshold as low as 0.5 already clears sub-1% error while accepting
~99% of fields — the scores are so well-behaved that almost no threshold is needed at all.
On **T2**, sub-1% error needs roughly 0.9, and only accepts ~4-7% of fields at that bar. On
**T3**, no threshold in the sweep reaches sub-1% error — the miscalibration from Finding 2
means there is no safe cutoff on badly degraded documents, only a trade-off between coverage
and risk.
"""
)

md(
    """
## 10. Finding 4 — Do the Shipped Thresholds Survive?

The cookbook ships 0.85 and 0.97. Applied to each tier:
"""
)

code(
    """
Image.open(RESULTS / "figures" / "04_fixed_thresholds.png")
"""
)

code(
    """
pd.read_csv(RESULTS / "fixed_thresholds.csv")
"""
)

md(
    """
**0.85 is safe on T0 and T1 (0% error) and unsafe everywhere else** — 9% error on T2, over
40% on T3. **0.97 recovers safety but at a steep coverage cost**, accepting only ~1% of
fields on T2 and T3. Neither threshold is wrong; each is correct for a document quality it
was never labelled as being specific to. Treating 0.85 as a universal bar, the way the
`unsiloed` skill currently does, is the actual bug — not the number itself.
"""
)

md(
    """
## 11. What This Suggests

- **Route by document quality, not just by score.** A scanned or photographed document
  needs a stricter threshold than a clean PDF at the *same* nominal confidence — the number
  means less the worse the source image is.
- **`grounding_score` and `extraction_score` can be used interchangeably** for the
  accept/reject decision (Finding 1); the meaningful choice is the threshold and how it
  varies by document condition, not which of the two scores to read.
- **A single global default (0.85 or 0.97) cannot be correct for every input.** The
  cookbook's own skills should probably ask what kind of document is being processed before
  picking one.

## Limitations

- **The generated corpus is synthetic.** Exact ground truth and a controlled difficulty
  ladder are not possible with real, unlabelled documents — this is the trade made to get
  them. The degradation tiers were tuned by eye: T3 is meant to be genuinely hard but still
  legible to a person (see the sample above); a tier no one could read would be measuring an
  impossible task, not calibration.
- **CORD receipts are real but narrow** — Indonesian retail receipts, thermal-printed. They
  validate that miscalibration on hard documents is a real phenomenon, not an artefact of
  synthetic degradation, but the specific numbers may not transfer to, say, financial filings.
- **Thresholds here are specific to this corpus.** The methodology — generate, degrade,
  measure — is the reusable part; a team should run it against their own documents rather
  than adopt these exact numbers.
- **Scores move between runs** as Unsiloed's models change in production. Treat the figures
  as indicative of the *shape* of the finding (thresholds do not travel across document
  quality) rather than as fixed targets.

## Reproducing This

```bash
cd confidence-calibration
python -m corpus.generate     # 12 documents, ~5s
python -m corpus.degrade      # four tiers, ~10s
python -m corpus.cord         # 92 receipts, ~3min, needs no API key
python -m calibration.run     # 140 extractions, ~5min, needs UNSILOED_API_KEY
python -m calibration.analyse # figures + tables, ~30s, no API calls
```

## Related

- [Unsiloed extract API reference](https://docs.unsiloed.ai/api-reference/extraction/extract-data)
- [Confidence Score Reliability: The Missing Metric](https://www.unsiloed.ai/blog/confidence-score-reliability-the-missing-metric-in-document-extraction)
- [Claude + Unsiloed tool use](../claude/tool-use/) and [Gemini function calling](../gemini/function-calling/) — the two scores this notebook measures were first described there
"""
)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
