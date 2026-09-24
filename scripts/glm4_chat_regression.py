"""GLM-4-9B chat regression: the tokenizer bridge around the C++ engine.

WHY A BRIDGE
------------
The engine's own TokenizerFactory cannot serve this checkpoint, and the gap is
structural rather than a missing feature flag:

  * it requires `tokenizer.json` (a byte-level-BPE dump with model.vocab +
    model.merges) and throws when the file is absent. GLM-4 ships only
    `tokenizer.model` -- a tiktoken-format rank file driven by a CUSTOM tokenizer
    class, ChatGLM4Tokenizer, in `tokenization_chatglm.py`;
  * its ChatTemplateFactory recognizes exactly two families, ChatML
    (`<|im_start|>`) and Llama-3 (`<|start_header_id|>`), and throws on anything
    else. GLM-4's template is a third family: `[gMASK]<sop>` then
    `<|role|>\\n{content}` per message.

Rather than block a model regression on porting a tiktoken BPE and a third
template family into C++, this script owns tokenization and detokenization and
hands the engine nothing but token ids. That is a deliberately narrow bridge: the
ids crossing it are the same ids the engine would have produced natively, so the
generation being measured is entirely the engine's.

PROTOCOL
--------
  --emit    writes jobs.tsv (token ids per turn) + prompts.txt (what they render as)
  --run     also invokes the C++ runner, then decodes  (the usual one-shot)
  --decode  reads outputs.tsv and prints the verbatim report

jobs.tsv, one line per conversation turn:
    case <TAB> turn <TAB> reset <TAB> max_new <TAB> id,id,id,...
`reset=1` restarts the sequence at position 0; `reset=0` APPENDS to the running KV
cache, which is what makes turn 2 of the memory case a cache-invariance test rather
than a re-prefill.

Usage:
    python scripts/glm4_chat_regression.py --run
    python scripts/glm4_chat_regression.py --emit --out-dir <dir>
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import io
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TOKENIZER_DIR = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI/models"), "GLM-4-9B-Chat-1M-hf")
DEFAULT_OUT_DIR = REPO_ROOT / "out" / "glm4_chat_regression"
# RELEASE by default: the Debug CUDA build compiles kernels with -G -Od, which makes
# any tokens/sec figure a measurement of the build flags rather than of the engine
# (measured ~1.45 tok/s in Debug vs the Release number in the report). Debug is only
# used if no Release runner has been built.
_RUNNERS = [
    REPO_ROOT / "out" / "build" / "x64-Release" / "tests" / "Release" / "integration_tests.exe",
    REPO_ROOT / "out" / "build" / "x64-Debug" / "tests" / "Debug" / "integration_tests.exe",
]
DEFAULT_RUNNER = next((p for p in _RUNNERS if p.exists()), _RUNNERS[0])

# The four regression cases. `turns` is a list of user messages; every case after
# the first turn appends to the SAME KV cache.
CASES = [
    ("identity", 200, [
        "Who are you and what model architecture are you based on? "
        "Answer in one short paragraph.",
    ]),
    ("arithmetic", 320, [
        "Solve this step by step: A box contains 15 red balls and 27 blue balls. "
        "If I remove 8 red balls and add 12 blue balls, how many total balls are "
        "in the box?",
    ]),
    ("coding", 320, [
        "Write a Python function 'fibonacci(n: int) -> list[int]' that returns the "
        "first n Fibonacci numbers. Include type hints and handle n <= 0.",
    ]),
    ("memory", 120, [
        "Remember this secret code word: PINEAPPLE-42.",
        "What was the secret code word I asked you to remember?",
    ]),
]


def load_tokenizer(tokenizer_dir: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)


class Glm4Prompt:
    """Builds GLM-4 chat turns as explicit id sequences.

    The template (tokenizer_config.json) is:
        [gMASK]<sop>{ for each message }<|{role}|>{metadata}\\n{content}{ end }
        { if add_generation_prompt }<|assistant|>
    Note there is NO newline after the generation prompt -- the model emits it
    itself, which is why a well-formed GLM-4 reply starts with token 198 ("\\n").

    Pieces are converted separately (special tokens by id, text with
    add_special_tokens=False) so that a DELTA for turn 2+ carries no [gMASK]<sop>
    preamble. `verify_against_template` checks the construction against the
    tokenizer's own apply_chat_template, so a drift in either is caught here.
    """

    def __init__(self, tok):
        self.tok = tok
        self.id = {name: tok.convert_tokens_to_ids(name) for name in
                   ("[gMASK]", "<sop>", "<|user|>", "<|assistant|>",
                    "<|system|>", "<|observation|>", "<|endoftext|>")}
        missing = [k for k, v in self.id.items() if v is None or v < 0]
        if missing:
            raise RuntimeError(f"tokenizer is missing special tokens: {missing}")

    def _text(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def first_turn(self, user_msg: str) -> list[int]:
        return ([self.id["[gMASK]"], self.id["<sop>"], self.id["<|user|>"]] +
                self._text("\n" + user_msg) + [self.id["<|assistant|>"]])

    def next_turn(self, user_msg: str) -> list[int]:
        """The delta appended AFTER the previous turn's generated reply."""
        return ([self.id["<|user|>"]] + self._text("\n" + user_msg) +
                [self.id["<|assistant|>"]])

    def verify_against_template(self, user_msg: str) -> tuple[bool, list[int], list[int]]:
        mine = self.first_turn(user_msg)
        theirs = self.tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            add_generation_prompt=True, tokenize=True)
        # ChatGLM4Tokenizer returns a BatchEncoding, which is a UserDict -- a
        # Mapping but NOT a dict instance, so isinstance(_, dict) misses it.
        if isinstance(theirs, Mapping):
            theirs = theirs["input_ids"]
        return mine == list(theirs), mine, list(theirs)

    def stop_ids(self) -> list[int]:
        # generation_config.json: <|endoftext|>, <|user|>, <|observation|>
        return [self.id["<|endoftext|>"], self.id["<|user|>"], self.id["<|observation|>"]]


def emit(out_dir: Path, tok) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Glm4Prompt(tok)

    ok, mine, theirs = p.verify_against_template(CASES[0][2][0])
    if not ok:
        print("[error] manual turn construction disagrees with apply_chat_template:")
        print(f"        mine  : {mine}")
        print(f"        theirs: {theirs}")
        raise SystemExit(3)
    print(f"[emit] turn construction matches apply_chat_template "
          f"({len(mine)} ids for case 'identity')")

    jobs = io.open(out_dir / "jobs.tsv", "w", encoding="utf-8", newline="\n")
    pretty = io.open(out_dir / "prompts.txt", "w", encoding="utf-8", newline="\n")
    jobs.write(f"# stop_ids\t{','.join(str(i) for i in p.stop_ids())}\n")

    for case, max_new, turns in CASES:
        for t, msg in enumerate(turns):
            ids = p.first_turn(msg) if t == 0 else p.next_turn(msg)
            jobs.write(f"{case}\t{t}\t{1 if t == 0 else 0}\t{max_new}\t"
                       f"{','.join(str(i) for i in ids)}\n")
            pretty.write(f"===== case={case} turn={t} "
                         f"({'fresh sequence' if t == 0 else 'KV-append delta'}, "
                         f"{len(ids)} tokens, max_new={max_new}) =====\n")
            pretty.write(tok.decode(ids) + "\n\n")
    jobs.close()
    pretty.close()
    print(f"[emit] wrote {out_dir/'jobs.tsv'} and {out_dir/'prompts.txt'}")


def decode(out_dir: Path, tok) -> int:
    path = out_dir / "outputs.tsv"
    if not path.exists():
        print(f"[error] {path} not found -- run the C++ runner first")
        return 2

    rows = []
    for line in io.open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            if line.startswith("# telemetry"):
                print("\n" + line[2:])
            continue
        f = line.split("\t")
        rows.append(dict(case=f[0], turn=int(f[1]), ids=[int(x) for x in f[2].split(",") if x],
                         stop=f[3], prompt_tokens=int(f[4]), gen_tokens=int(f[5]),
                         prefill_ms=float(f[6]), decode_ms=float(f[7])))

    print("\n" + "=" * 78)
    print("GLM-4-9B-Chat-1M  --  verbatim generation output")
    print("=" * 78)
    for r in rows:
        text = tok.decode(r["ids"], skip_special_tokens=False)
        tps = (r["gen_tokens"] / (r["decode_ms"] / 1000.0)) if r["decode_ms"] > 0 else 0.0
        print(f"\n----- case={r['case']} turn={r['turn']} "
              f"(prompt {r['prompt_tokens']} tok, generated {r['gen_tokens']} tok, "
              f"stop={r['stop']}, {tps:.2f} tok/s) -----")
        print(text)
    print("\n" + "=" * 78)

    print("\nper-turn telemetry")
    print(f"{'case':<12}{'turn':>5}{'prompt':>8}{'gen':>6}{'prefill_ms':>12}"
          f"{'decode_ms':>11}{'tok/s':>8}  stop")
    tot_gen, tot_ms = 0, 0.0
    for r in rows:
        tps = (r["gen_tokens"] / (r["decode_ms"] / 1000.0)) if r["decode_ms"] > 0 else 0.0
        print(f"{r['case']:<12}{r['turn']:>5}{r['prompt_tokens']:>8}{r['gen_tokens']:>6}"
              f"{r['prefill_ms']:>12.1f}{r['decode_ms']:>11.1f}{tps:>8.2f}  {r['stop']}")
        tot_gen += r["gen_tokens"]
        tot_ms += r["decode_ms"]
    if tot_ms > 0:
        print(f"{'TOTAL':<12}{'':>5}{'':>8}{tot_gen:>6}{'':>12}{tot_ms:>11.1f}"
              f"{tot_gen / (tot_ms/1000.0):>8.2f}")

    # --- automated checks (the same ones the task statement asks for) ---------
    print("\nchecks")
    by = {(r["case"], r["turn"]): tok.decode(r["ids"], skip_special_tokens=True)
          for r in rows}
    stops = {(r["case"], r["turn"]): r["stop"] for r in rows}
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    ident = by.get(("identity", 0), "")
    check("identity mentions GLM", "GLM" in ident or "ChatGLM" in ident)
    check("identity halted on a stop token", stops.get(("identity", 0)) == "stop_token",
          stops.get(("identity", 0), "?"))

    arith = by.get(("arithmetic", 0), "")
    check("arithmetic arrives at 46", "46" in arith)
    check("arithmetic shows step-by-step work",
          sum(tok_ in arith for tok_ in ("15", "27", "8", "12")) >= 3)

    code = by.get(("coding", 0), "")
    check("coding defines fibonacci with type hints",
          "def fibonacci" in code and "list[int]" in code)
    check("coding handles n <= 0", ("n <= 0" in code or "n<=0" in code or
                                   "n < 1" in code or "n <= 0" in code))
    if "def fibonacci" in code:
        start = code.index("def fibonacci")
        snippet = code[start:]
        for fence in ("```", "\n\n\n"):
            if fence in snippet:
                snippet = snippet[:snippet.index(fence)]
        try:
            compile(snippet, "<generated>", "exec")
            check("coding snippet compiles as Python", True)
        except SyntaxError as e:
            check("coding snippet compiles as Python", False, f"{e.msg} line {e.lineno}")
    else:
        check("coding snippet compiles as Python", False, "no function definition found")

    recall = by.get(("memory", 1), "")
    check("multi-turn recalls PINEAPPLE-42", "PINEAPPLE-42" in recall.upper())

    print(f"\n{len(failures)} check(s) failed" if failures else "\nall checks passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--runner", default=str(DEFAULT_RUNNER))
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="emit, invoke the C++ runner, then decode")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not (args.emit or args.decode or args.run):
        args.run = True

    tok = load_tokenizer(args.tokenizer_dir)

    if args.emit or args.run:
        emit(out_dir, tok)
    if args.run:
        env = dict(os.environ, BLACKWELL_GLM4_CHAT_DIR=str(out_dir))
        cmd = [args.runner, "--gtest_filter=Glm4ChatRegression.*", "--gtest_color=no"]
        print(f"[run] {' '.join(cmd)}")
        rc = subprocess.call(cmd, env=env)
        print(f"[run] runner exit {rc}")
    if args.decode or args.run:
        return decode(out_dir, tok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
