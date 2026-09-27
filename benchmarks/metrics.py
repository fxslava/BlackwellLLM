"""Self-contained LongBench scoring metrics.

Faithful reimplementation of the scorers in THUDM/LongBench's `metrics.py` +
`eval.py`, with **no third-party dependencies** -- this machine has neither
`fuzzywuzzy`/`rapidfuzz` nor `rouge` installed, and a benchmark baseline that
cannot be re-scored later because a package is missing is not a baseline.

What is deliberately identical to the official implementation:

* `normalize_answer` -- lowercase, strip punctuation, drop the articles a/an/the,
  collapse whitespace (SQuAD's normalisation, which LongBench inherits).
* `qa_f1_score`   -- token-level F1 over the normalised strings.
* `classification_score` -- the substring-voting rule used for `trec`, including
  its removal of class names that are a *proper* substring of the gold label.
* `code_sim_score` -- first line of the prediction that is not a comment, scored
  against the gold line by edit-similarity.
* `scorer` -- per-task dispatch, `max` over multiple gold answers, and the
  `lstrip('\\n').split('\\n')[0]` first-line postprocessing that the official
  script applies to the classification/summarisation family (`trec` included).

The one place where a *choice* had to be made is the edit similarity, because the
official script's `fuzz.ratio` is not a single well-defined function: `fuzzywuzzy`
uses `python-Levenshtein`'s Indel ratio when the C extension is installed and
falls back to `difflib`'s Ratcliff-Obershelp ratio when it is not, and the two
disagree by a point or two on short strings. `code_sim_score` here uses the
**Indel (Levenshtein) ratio**, i.e. `2*LCS/(len(a)+len(b))`, which is what
`python-Levenshtein.ratio` and `rapidfuzz.fuzz.ratio` compute -- the brief asks
for Levenshtein edit similarity, and it is the LCS-optimal of the two.
`difflib_ratio` is provided alongside so a run can report the gap.
"""

from __future__ import annotations

import difflib
import re
import string
from collections import Counter
from typing import Callable, Iterable, Sequence

__all__ = [
    "normalize_answer",
    "qa_f1_score",
    "rouge_l_score",
    "code_sim_score",
    "code_exact_prefix_score",
    "classification_score",
    "indel_ratio",
    "difflib_ratio",
    "TASK_METRIC",
    "score_sample",
    "scorer",
]


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation, as LongBench uses it."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


# --------------------------------------------------------------------------- #
# token-level F1
# --------------------------------------------------------------------------- #

def _f1(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Token-level F1 after normalisation. The metric for qasper / multifieldqa_en."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    return _f1(pred_tokens, gold_tokens)


# --------------------------------------------------------------------------- #
# ROUGE-L
# --------------------------------------------------------------------------- #

def _lcs_length(a: Sequence, b: Sequence) -> int:
    """Length of the longest common subsequence, O(len(a)*len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Standard ROUGE-L F-measure (beta = 1) over normalised word sequences.

    Reported as a secondary diagnostic: none of the four tasks in this baseline
    score on ROUGE officially, but it separates "answered with the right content
    in the wrong order/length" from "missed the fact", which token-F1 blurs.
    Note this is the beta=1 harmonic mean, NOT the `rouge` package's beta=P/R
    variant -- the two differ whenever precision and recall are far apart.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# edit similarity
# --------------------------------------------------------------------------- #

def indel_ratio(a: str, b: str) -> float:
    """Indel (Levenshtein) similarity in [0, 1]: 2*LCS(a,b) / (len(a)+len(b)).

    Equal to `python-Levenshtein.ratio` / `rapidfuzz.fuzz.ratio` / 100 -- the
    Levenshtein distance restricted to insertions and deletions (substitution
    cost 2), normalised by the summed length.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2.0 * _lcs_length(a, b) / (len(a) + len(b))


def difflib_ratio(a: str, b: str) -> float:
    """Ratcliff-Obershelp ratio -- what `fuzzywuzzy` falls back to with no C extension.

    Kept so a report can quantify the metric-implementation gap rather than
    asserting the two are interchangeable.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _first_code_line(prediction: str) -> str:
    """The official rule: first line that is not a comment / markdown fence."""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return line
    return ""


def code_sim_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Edit similarity of the predicted next line vs the gold line. Metric for lcc."""
    return indel_ratio(_first_code_line(prediction), ground_truth)


def code_exact_prefix_score(prediction: str, ground_truth: str, **_: object) -> float:
    """1.0 when the predicted next line matches the gold line exactly (whitespace-trimmed).

    The brief's "exact match prefix" companion to edit similarity: edit-sim near
    90 can still mean every line is subtly wrong, and this separates the two.
    Trailing/leading whitespace is ignored because indentation is scored by the
    edit similarity, not here.
    """
    return 1.0 if _first_code_line(prediction).strip() == ground_truth.strip() else 0.0


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #

def classification_score(prediction: str, ground_truth: str,
                         all_classes: Iterable[str] | None = None, **_: object) -> float:
    """LongBench's substring-voting rule for trec.

    Every class name occurring in the prediction is a vote; a vote that is a
    *proper* substring of the gold label is dropped (so "location" does not also
    fire for "location:city"); the score is 1/len(votes) when the gold label is
    among them, else 0. Reproduces the official function, including its lenience
    towards a prediction that names several classes.
    """
    if not all_classes:
        return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
    votes = [c for c in all_classes if c in prediction]
    votes = [c for c in votes if not (c in ground_truth and c != ground_truth)]
    if ground_truth in votes:
        return 1.0 / len(votes)
    return 0.0


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

# Official LongBench task -> metric map, restricted to the tasks this harness runs.
TASK_METRIC: dict[str, Callable[..., float]] = {
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "lcc": code_sim_score,
    "trec": classification_score,
}

# Tasks whose prediction is cut to its first line before scoring (official eval.py).
FIRST_LINE_ONLY = {"trec", "triviaqa", "samsum", "lsht"}

# Secondary metrics recorded per sample for failure-mode analysis. They are NOT
# the task's official score; the report must quote `score` for that.
SECONDARY_METRIC: dict[str, dict[str, Callable[..., float]]] = {
    "qasper": {"rouge_l": rouge_l_score},
    "multifieldqa_en": {"rouge_l": rouge_l_score},
    "lcc": {"exact_prefix": code_exact_prefix_score,
            "edit_sim_difflib": lambda p, g, **k: difflib_ratio(_first_code_line(p), g)},
    "trec": {},
}


def _postprocess(task: str, prediction: str) -> str:
    if task in FIRST_LINE_ONLY:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def score_sample(task: str, prediction: str, ground_truths: Sequence[str],
                 all_classes: Iterable[str] | None = None) -> dict[str, float]:
    """Official score plus the secondary diagnostics, for one sample.

    `max` over the gold answers, exactly as the official scorer does.
    """
    metric = TASK_METRIC[task]
    pred = _postprocess(task, prediction)
    out = {"score": max((metric(pred, g, all_classes=all_classes) for g in ground_truths),
                        default=0.0)}
    for name, fn in SECONDARY_METRIC.get(task, {}).items():
        out[name] = max((fn(pred, g, all_classes=all_classes) for g in ground_truths),
                        default=0.0)
    return out


def scorer(task: str, predictions: Sequence[str], answers: Sequence[Sequence[str]],
           all_classes: Iterable[str] | None = None) -> float:
    """Task score on LongBench's 0-100 scale, rounded to 2 dp as the official script does."""
    if not predictions:
        return 0.0
    total = sum(score_sample(task, p, g, all_classes)["score"]
                for p, g in zip(predictions, answers))
    return round(100 * total / len(predictions), 2)


# --------------------------------------------------------------------------- #
# self-test -- `python metrics.py` verifies the scorers against hand-computed values
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    failures = []

    def check(name, got, want, tol=1e-9):
        if abs(got - want) > tol:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    check("normalize/articles",
          1.0 if normalize_answer("The Cat, a Dog!") == "cat dog" else 0.0, 1.0)

    # F1: pred {x,y,z} vs gold {y,z,w} -> 2 common, P=R=2/3, F1=2/3.
    # NB the tokens avoid a/an/the on purpose -- normalize_answer strips articles,
    # so "a b c" would score as the 2-token "b c" and this check would be testing
    # article removal instead of F1.
    check("qa_f1 partial", qa_f1_score("x y z", "y z w"), 2 / 3)
    check("qa_f1 exact", qa_f1_score("the answer", "answer"), 1.0)
    check("qa_f1 disjoint", qa_f1_score("xyz", "abc"), 0.0)
    check("qa_f1 empty pred", qa_f1_score("", "abc"), 0.0)

    # ROUGE-L: "w x y z" vs "w y z" -> LCS=3, P=3/4, R=1 -> F=2*.75*1/1.75
    check("rouge_l", rouge_l_score("w x y z", "w y z"), 2 * 0.75 * 1.0 / 1.75)
    # order matters for LCS where bag-of-tokens F1 cannot see it
    check("rouge_l order", rouge_l_score("z y x", "x y z"), 2 * (1/3) * (1/3) / (2/3))

    # Indel ratio: "abcd" vs "abed" -> LCS=3, 2*3/8 = 0.75
    check("indel", indel_ratio("abcd", "abed"), 0.75)
    check("indel identical", indel_ratio("x", "x"), 1.0)
    check("indel empty", indel_ratio("", "x"), 0.0)

    # code_sim skips a comment line, then scores the first real line
    check("code_sim skips comment",
          code_sim_score("# a comment\n    return a + b", "    return a + b"), 1.0)
    check("code_exact_prefix",
          code_exact_prefix_score("\n  return a + b  ", "return a + b"), 1.0)
    check("code_exact_prefix negative",
          code_exact_prefix_score("return a - b", "return a + b"), 0.0)

    classes = ["location", "location:city", "number"]
    # Prediction names the gold label only -> 1.0
    check("classification single", classification_score("location:city", "location:city",
                                                        all_classes=classes), 1.0)
    # "number" is an unrelated extra vote -> 1/2 ("location" is dropped as a proper
    # substring of the gold label)
    check("classification split", classification_score("location:city and number",
                                                       "location:city",
                                                       all_classes=classes), 0.5)
    check("classification miss", classification_score("number", "location:city",
                                                      all_classes=classes), 0.0)

    # scorer: 0-100 scale, max over golds, first-line-only for trec
    check("scorer trec first line",
          scorer("trec", ["number\nlocation:city"], [["number"]], classes), 100.0)
    check("scorer qa max over golds",
          scorer("qasper", ["blue"], [["red", "blue"]]), 100.0)

    if failures:
        print("SELFTEST FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("metrics.py selftest: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
