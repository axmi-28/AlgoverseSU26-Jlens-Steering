"""Domain corpora, for conditioning the average Jacobian on an input distribution.

The lens ships one ``J_bar`` fitted on wikitext. This module supplies the
corpora needed to ask what that choice bought: compute ``J_x`` over a *different*
input distribution, average within it, and compare the resulting ``J_bar_domain``
against the published one and against each other.

Why this is a real question rather than a fishing trip
------------------------------------------------------
The J-lens paper equivocates. Its Sec. 2 says averaging isolates "the model's
general disposition to verbalize a given concept" -- a property of the model --
but also that ``J_bar`` measures effects "across the range of contexts the model
encounters" -- a property of a distribution. It never resolves which, and its
A.7 "Distribution" paragraph varies the corpus only by *restricting or masking
within the same pretraining-like distribution* ("excluding the first several
tokens", "excluding positions whose next token is non-alphanumeric"), never a
genuinely different domain, and scores only readout quality.

A follow-up (arXiv 2608.25347, Thm. 1) makes the prediction sharp: for Gaussian
``h``, ``E[J_f(h)] = Cov(f(h), h) Cov(h)^-1``, the population least-squares
slope. The covariance factor is the *corpus* covariance, so two corpora with
different second moments cannot share an averaged Jacobian. Under that identity
a domain split is a test of a theorem, not an exploration -- which is also why
the geometry alone is a weak headline and the causal arm matters.

The three-corpus design, and why not simply "chat vs math"
-----------------------------------------------------------
Chat is a *format* (turn structure, an instruct template, second-person
register); math is a *topic*. Contrasting them directly moves both at once, and
a difference would not say which mattered. So each corpus here moves one
variable against the same wikitext control:

- ``wikitext_a`` / ``wikitext_b`` -- disjoint halves of the fitting corpus. Not
  a corpus of interest: the **null**. Every statistic computed between two
  domains is also computed between these two, at identical N, and only the
  excess over that floor is evidence. ``J_bar`` is itself a mean over this
  distribution, so any displacement measured here is finite-sample error by
  construction. This is the calibration the whole comparison rests on.
- ``openwebmath`` -- topic moves, prose format held constant against wikitext.
- ``ultrachat`` -- register and turn structure move, topic stays general. Kept
  as **raw text**, not wrapped in the instruct template, so that this arm
  differs from wikitext in discourse and not in the presence of control tokens;
  a templated arm would confound the two and belongs in a separate condition.

Length is matched, not incidental
----------------------------------
Every prompt here is filtered to at least ``max_seq_len`` tokens and truncated
to it, so all four groups present the model with the same number of positions
under the same ``skip_first``. Without that, "domain" is confounded with
sequence length -- which changes both the position set the estimator averages
over and the cost of the prompt (a 128-token prompt is ~16x an 8-token one).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Cheap pre-filter; the real filter is the token count. Documents shorter than
#: this cannot reach 128 tokens under any tokenizer worth using.
_MIN_CHARS = 256


@dataclass(frozen=True)
class DomainSpec:
    """One corpus, pinned tightly enough to be reproducible.

    Attributes:
        name: The group label, and the prefix of every prompt key it yields.
        dataset / config / split / text_field: HuggingFace coordinates.
        flatten_messages: Read ``text_field`` as a chat-style list of
            ``{"role", "content"}`` turns and join their contents, rather than
            as a string. UltraChat stores conversations this way.
        stride / offset: Take every ``stride``-th document starting at
            ``offset``. This is what makes ``wikitext_a`` and ``wikitext_b``
            disjoint *and* interleaved: they read the same region of the same
            stream, so any drift in dataset ordering is shared between them
            rather than being a difference the null would absorb.
    """

    name: str
    dataset: str
    config: str | None
    split: str
    text_field: str = "text"
    flatten_messages: bool = False
    stride: int = 1
    offset: int = 0
    #: Second field appended after a newline. GSM8K keeps the problem in
    #: ``question`` and the worked solution in ``answer``; the solution is where
    #: the arithmetic actually happens, so a question-only document would be
    #: word problems with the maths removed.
    join_field: str | None = None
    #: Regexes deleted from each document, for dataset-specific markup that is
    #: an artifact of the release rather than of the domain.
    strip_patterns: tuple[str, ...] = ()
    #: Name of a generator in :data:`GENERATORS` to synthesize documents from,
    #: instead of streaming a HuggingFace dataset.
    generator: str | None = None
    #: Concatenate consecutive records until the document reaches
    #: ``min_tokens``, instead of discarding short ones.
    #:
    #: Needed for the word-problem corpora. One SVAMP item is ~50 tokens, so a
    #: 128-token filter rejects every record and the corpus silently yields
    #: nothing -- or, worse, yields only the handful of freak-long items, which
    #: is a biased sample rather than an empty one. Packing keeps the sample
    #: representative at the cost of putting ~3 problems in a document, which
    #: is what GSM8K's question+solution already looks like at this length.
    pack: bool = False
    #: Extra fields appended after ``join_field``, in order. AQuA-RAT needs
    #: question + rationale; SVAMP needs body + question + equation.
    extra_fields: tuple[str, ...] = ()
    #: Name of a function in :data:`TRANSFORMS` applied to each document after
    #: assembly. These are the within-corpus ablations: same source text, one
    #: structural property destroyed, so topic and vocabulary are held fixed
    #: and only the structure varies. A corpus comparison across *datasets*
    #: can never do that -- GSM8K and wikitext differ in a hundred ways at
    #: once -- which is why these are the load-bearing controls.
    transform: str | None = None

    def identity(self) -> str:
        """Everything about this corpus that changes the documents it yields."""
        return "|".join(
            [
                self.name,
                self.dataset,
                self.config or "",
                self.split,
                self.text_field,
                str(self.flatten_messages),
                f"{self.offset}/{self.stride}",
                self.join_field or "",
                ";".join(self.strip_patterns),
                self.generator or "",
                str(self.pack),
                ";".join(self.extra_fields),
                self.transform or "",
            ]
        )


#: The default panel. Order is fixed because it becomes the prompt list that
#: every shard strides.
#:
#: **Four wikitext quarters, not two halves.** With two, the null is a single
#: number per layer and there is no way to tell a real domain effect from one
#: null draw landing low. Four disjoint quarters give six same-distribution
#: pairs, so the null becomes a *range* and a domain pair is evidence only if
#: it falls outside it. They are strided rather than blocked so that all four
#: read the same region of the stream and share any drift in dataset ordering,
#: which would otherwise show up as a difference the null absorbs.
DEFAULT_DOMAINS: tuple[DomainSpec, ...] = (
    DomainSpec(
        "wikitext_a",
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "train",
        stride=4,
        offset=0,
    ),
    DomainSpec(
        "wikitext_b",
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "train",
        stride=4,
        offset=1,
    ),
    DomainSpec(
        "wikitext_c",
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "train",
        stride=4,
        offset=2,
    ),
    DomainSpec(
        "wikitext_d",
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "train",
        stride=4,
        offset=3,
    ),
    DomainSpec("openwebmath", "open-web-math/open-web-math", None, "train"),
    DomainSpec(
        "gsm8k",
        "openai/gsm8k",
        "main",
        "train",
        text_field="question",
        join_field="answer",
        # The calculator markup and answer sentinel are release artifacts, not
        # features of mathematical text; leaving them in would move Sigma for a
        # reason unrelated to the domain.
        strip_patterns=(r"<<[^>]*>>", r"\n####.*$"),
    ),
    DomainSpec(
        "arith_words",
        "synthetic:arithmetic_words",
        None,
        "train",
        generator="arithmetic_words",
    ),
    DomainSpec(
        "ultrachat",
        "HuggingFaceH4/ultrachat_200k",
        None,
        "train_sft",
        text_field="messages",
        flatten_messages=True,
    ),
)

DOMAINS_BY_NAME = {spec.name: spec for spec in DEFAULT_DOMAINS}


#: Number of disjoint replicates registered per corpus for the Tier-0 spread.
N_REPLICATES = 5


def replicate_specs(base: str, n: int, *, prefix: str = "") -> tuple[DomainSpec, ...]:
    """``n`` disjoint draws from one corpus, for putting an error bar on it.

    Every domain result so far rests on a single draw of 32 documents, so a
    difference between two corpora and a difference between two draws of the
    same corpus are not distinguishable. These replicates are what separates
    them: fit the same pipeline on ``n`` non-overlapping samples and the spread
    across them is the null against which a corpus gap has to clear.

    Disjointness is by stride, not by slicing consecutive blocks. Strided draws
    all read the same region of the stream, so ordering drift inside the
    dataset is shared across replicates instead of being charged to one of
    them -- the same reason the four wikitext quarters are strided.

    A replicate's stride multiplies the base spec's, so replicating a corpus
    that is already strided stays disjoint from its siblings.
    """
    spec = DOMAINS_BY_NAME[base]
    return tuple(
        DomainSpec(
            name=f"{prefix or base}_r{i}",
            dataset=spec.dataset,
            config=spec.config,
            split=spec.split,
            text_field=spec.text_field,
            flatten_messages=spec.flatten_messages,
            stride=spec.stride * n,
            offset=spec.offset + i * spec.stride,
            join_field=spec.join_field,
            strip_patterns=spec.strip_patterns,
            generator=spec.generator,
        )
        for i in range(n)
    )


# --------------------------------------------------------------------------
# synthetic corpora
# --------------------------------------------------------------------------

_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}


def _word(n: int) -> str:
    """Small integers as English words -- the surface form the eval grades in."""
    if n in _WORDS:
        return _WORDS[n]
    tens, ones = divmod(n, 10)
    if 2 <= tens <= 9:
        base = {
            2: "twenty",
            3: "thirty",
            4: "forty",
            5: "fifty",
            6: "sixty",
            7: "seventy",
            8: "eighty",
            9: "ninety",
        }[tens]
        return base if ones == 0 else f"{base}-{_WORDS[ones]}"
    return str(n)


def arithmetic_word_documents(n: int, *, seed: int = 0) -> list[str]:
    """Simple arithmetic stated in number words, one fact per sentence.

    This corpus exists to be the *most favourable possible* averaging
    distribution for the numbers eval: same register, same vocabulary, same
    single-step computation on small integers, and crucially the same surface
    form -- number words rather than digits, which is what the eval grades and
    what GSM8K does not provide.

    Two deliberate restrictions, both of which protect the experiment:

    * **Arithmetic only.** No orthographic statements ("the word 'six' begins
      with s"). The eval's ``first_letter`` template is the control separating
      "helps arithmetic computation" from "makes the argument swap land harder
      in general", and seeding this corpus with orthography would destroy that
      contrast.
    * **No eval template appears verbatim.** These phrasings are deliberately
      different surface forms from the four eval templates, so the corpus is
      relevant by construction rather than by containing the test items. It
      shares vocabulary and computation type, not wording.

    Report this arm as the strongest form of the hypothesis: if averaging over
    text this close to the eval does not improve steering, weaker notions of
    domain relevance are very unlikely to.
    """
    import random

    rng = random.Random(seed)
    phrasings = [
        lambda a, b: (
            f"{_word(a).capitalize()} multiplied by {_word(b)} gives {_word(a * b)}."
        ),
        lambda a, b: f"The product of {_word(a)} and {_word(b)} is {_word(a * b)}.",
        lambda a, b: f"Adding {_word(a)} to {_word(b)} makes {_word(a + b)}.",
        lambda a, b: f"The sum of {_word(a)} and {_word(b)} is {_word(a + b)}.",
        lambda a, b: (
            f"Taking {_word(min(a, b))} away from {_word(max(a, b))} "
            f"leaves {_word(max(a, b) - min(a, b))}."
        ),
        lambda a, b: (
            f"One more than {_word(a)} is {_word(a + 1)}, and one less than "
            f"{_word(b)} is {_word(b - 1)}."
        ),
        lambda a, b: f"{_word(a).capitalize()} doubled is {_word(2 * a)}.",
    ]
    documents: list[str] = []
    for _ in range(n):
        # ~22 sentences clears 128 tokens with room to truncate.
        documents.append(
            " ".join(
                rng.choice(phrasings)(rng.randint(1, 12), rng.randint(1, 12))
                for _ in range(22)
            )
        )
    return documents


_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_ORDINALS = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
]


def ordered_scale_documents(n: int, *, seed: int = 0) -> list[str]:
    """Natural-language traversal of ordered scales, with NO arithmetic.

    The discriminating corpus for the ordered-scale hypothesis. ``arith_words``
    is its mirror image: arithmetic on integers with no other ordered scale;
    this is days, months and ordinals traversed in natural language with no
    computation at all. Between them they separate the two readings of the
    GSM8K result:

    * if steering improves on *numbers* prompts here, what transfers is
      traversal of an ordered scale, and the arithmetic in GSM8K is incidental;
    * if it improves only on weekday/month prompts, the effect is tied to the
      particular scale the corpus traverses;
    * if it improves nothing, then neither ingredient alone is sufficient and
      GSM8K's combination of the two is doing the work.

    As with ``arith_words``, no eval template appears verbatim -- these are
    deliberately different surface forms -- so the corpus is relevant by
    construction rather than by containing the test items.
    """
    import random

    rng = random.Random(seed)

    # (template, scale, gap). Templates are data rather than lambdas so the
    # corpus is inspectable: the whole claim rests on what this text contains.
    phrasings = [
        ("{0} is followed immediately by {1}.", _DAYS, 1),
        ("{0} comes two days before {1}.", _DAYS, 2),
        ("Counting forward from {0} brings you to {1}.", _DAYS, 1),
        ("{0} falls earlier in the sequence of months than {1}.", _MONTHS, 1),
        ("The month right after {0} is {1}.", _MONTHS, 1),
        ("{0} precedes {1} by two places.", _MONTHS, 2),
        ("The {0} entry is followed by the {1}.", _ORDINALS, 1),
        ("Whatever is {0} comes before whatever is {1}.", _ORDINALS, 2),
        ("Moving one place on from the {0} gives the {1}.", _ORDINALS, 1),
    ]
    documents: list[str] = []
    for _ in range(n):
        sentences = []
        # ~24 sentences clears 128 tokens with room to truncate.
        for _ in range(24):
            template, scale, gap = rng.choice(phrasings)
            i = rng.randrange(len(scale) - gap)
            sentences.append(template.format(scale[i], scale[i + gap]))
        documents.append(" ".join(sentences))
    return documents


def equation_documents(n: int, *, seed: int = 0) -> list[str]:
    """Bare symbolic equations, no prose at all.

    The complement of strip_equations, and of arith_words -- which stated the
    same arithmetic in WORDS ("Seven doubled is fourteen") and scored 0/39. If
    symbols are what matter, this should work where arith_words did not, and
    the pair isolates notation from content.
    """
    import random

    rng = random.Random(seed)
    documents = []
    for _ in range(n):
        lines = []
        for _ in range(40):
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            op = rng.choice("+-*")
            val = {"+": a + b, "-": a - b, "*": a * b}[op]
            lines.append(f"{a} {op} {b} = {val}")
        documents.append("\n".join(lines))
    return documents


#: Generators addressable from a :class:`DomainSpec`.
GENERATORS = {
    "arithmetic_words": arithmetic_word_documents,
    "equations": equation_documents,
    "ordered_scale": ordered_scale_documents,
}


# --------------------------------------------------------------------------
# within-corpus ablations
# --------------------------------------------------------------------------
#
# Each destroys ONE property of a document and leaves the rest alone. Across
# datasets, "GSM8K works and wikitext does not" is compatible with a hundred
# explanations; within one dataset, "GSM8K works and sentence-shuffled GSM8K
# does not" is compatible with far fewer. These are deliberately crude: a
# transform that is easy to describe is easy to reason about when it fires.

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
#: Capitalised words that are not referents. Sentence openers and determiners
#: would break the grammar if replaced; months and weekdays are eval arguments
#: and must not be overwritten by the ablation.
_NOT_ENTITIES = frozenset(
    """the a an if in on at for and but so then when how what which that this
    these those there here it its is was were are be been being of to from with
    each every all some no not now after before during since until while
    january february march april may june july august september october
    november december monday tuesday wednesday thursday friday saturday sunday
    total each""".split()
)
_NAMES = (
    "Alice Bob Carol Dave Erin Frank Grace Heidi Ivan Judy Karl Lena "
    "Mallory Niaz Olga Peggy Quinn Rupert Sybil Trent Uma Victor Wendy"
).split()


def shuffle_sentences(text: str, *, seed: int = 0) -> str:
    """Same sentences, same tokens, derivation order destroyed.

    If the effect survives this, it is lexical -- the corpus supplies useful
    vocabulary or token statistics. If it dies, the ORDER of a worked solution
    is what matters, which no bag-of-tokens account of the averaged Jacobian
    would predict.
    """
    import random

    parts = [p for p in _SENTENCE.split(text) if p.strip()]
    random.Random(seed + len(text)).shuffle(parts)
    return " ".join(parts)


def scramble_entities(text: str, *, seed: int = 0) -> str:
    """Break coreference; keep sentence order and every number.

    Each *occurrence* of an entity mention gets a fresh random name, so an
    entity mentioned three times becomes three unrelated people. Sentence
    order and all arithmetic survive untouched.

    This separates "multi-step derivation" from "the same referent rebound as
    its value updates" -- and the latter is precisely what the swap operator
    does, which makes it the mechanistic hypothesis worth isolating.

    Entities are capitalised words plus personal pronouns. ``_NOT_ENTITIES``
    holds the capitalised words that are not referents -- sentence openers,
    months, weekdays -- because replacing those would wreck the grammar and
    confound "coreference broken" with "text degraded". Months and weekdays
    are excluded for a second reason: they are arguments of the eval, and
    injecting random names over them would change what the corpus says about
    exactly the tokens being steered.
    """
    import random

    rng = random.Random(seed + len(text))

    def swap(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.lower() in _NOT_ENTITIES:
            return word
        return rng.choice(_NAMES)

    text = re.sub(
        r"\b(?:[Hh]e|[Ss]he|[Tt]hey|him|her|them|his|hers|their)\b", swap, text
    )
    return re.sub(r"\b[A-Z][a-z]+\b", swap, text)


def strip_numbers(text: str) -> str:
    """Remove arithmetic, keep derivational prose.

    Digits become a placeholder, so the discourse structure of a worked
    solution survives with no computation left in it. The complement of
    arith_words, which is computation with no discourse.
    """
    return re.sub(r"\d[\d,.]*", "#", text)


def strip_equations(text: str) -> str:
    """Remove symbolic equality and arithmetic operators; keep every word.

    The surviving candidate after the first ablation round. Questions-only
    collapses to the control (6/105) while solutions-only does not (26/105),
    and the effect survives deleting every digit (35/105, unchanged), shuffling
    the sentences, and breaking coreference. What GSM8K *solutions* still have
    that its questions never did, and that all three of those ablations leave
    intact, is explicit symbolic assertion: "48 + 24 = 72" keeps its operators
    when the digits become "#", keeps them under shuffling, and keeps them when
    the names are scrambled.

    So this removes exactly that and nothing else. If the effect dies here, it
    is the symbolic equational form; if it survives, the remaining difference
    between questions and solutions is register alone.
    """
    return re.sub(r"\s*[=+*/<>^-]+\s*", " ", text)


#: Document-level ablations addressable from a :class:`DomainSpec`.
TRANSFORMS = {
    "strip_equations": strip_equations,
    "shuffle_sentences": shuffle_sentences,
    "scramble_entities": scramble_entities,
    "strip_numbers": strip_numbers,
}


def _document_text(record: dict, spec: DomainSpec) -> str | None:
    import re

    raw = record.get(spec.text_field)
    if raw is None:
        return None
    if not spec.flatten_messages:
        if not isinstance(raw, str):
            return None
        if spec.join_field:
            second = record.get(spec.join_field)
            if not isinstance(second, str):
                return None
            raw = f"{raw}\n{second}"
        for field in spec.extra_fields:
            extra = record.get(field)
            if extra is not None:
                raw = f"{raw}\n{extra}"
        for pattern in spec.strip_patterns:
            raw = re.sub(pattern, "", raw)
        raw = raw.strip()
        if spec.transform:
            raw = TRANSFORMS[spec.transform](raw)
        return raw or None
    # UltraChat: a list of {"role", "content"} turns. Joined with blank lines
    # and no role markers -- see the module docstring on keeping this arm raw.
    if not isinstance(raw, list):
        return None
    # Two conventions in the wild: UltraChat uses {"role", "content"},
    # OpenThoughts uses {"from", "value"}. Reading only "content" silently
    # yields an empty document for the latter, which surfaces as "0 documents
    # of >=128 tokens" rather than as a parse error.
    turns = [
        t.get("content") or t.get("value") or "" for t in raw if isinstance(t, dict)
    ]
    return "\n\n".join(t for t in turns if t) or None


def domain_prompts(
    spec: DomainSpec,
    *,
    tokenizer,
    n: int,
    max_chars: int = 2000,
    min_tokens: int = 128,
) -> Iterator[str]:
    """Yield ``n`` documents of at least ``min_tokens`` tokens, in dataset order.

    ``tokenizer`` is required, unlike :func:`jsteer.corpus.fitting_prompts`: the
    whole point of this panel is that the four groups present the same number
    of positions, and that cannot be checked without tokenizing.
    """
    if spec.generator:
        for text in GENERATORS[spec.generator](n * 2):
            if len(tokenizer.encode(text, add_special_tokens=False)) < min_tokens:
                continue
            yield text[:max_chars]
            n -= 1
            if n <= 0:
                return
        return

    from datasets import load_dataset

    dataset = load_dataset(spec.dataset, spec.config, split=spec.split, streaming=True)

    n_yielded = 0
    buffer: list[str] = []
    for index, record in enumerate(dataset):
        if (index - spec.offset) % spec.stride != 0 or index < spec.offset:
            continue
        text = _document_text(record, spec)
        if text is None:
            continue
        if spec.pack:
            # Accumulate whole records; never split one across documents, so a
            # packed document is always an integer number of problems.
            buffer.append(text)
            joined = "\n\n".join(buffer)[:max_chars]
            if len(tokenizer.encode(joined, add_special_tokens=False)) < min_tokens:
                continue
            buffer = []
            yield joined
        else:
            if len(text) < _MIN_CHARS:
                continue
            text = text[:max_chars]
            if len(tokenizer.encode(text, add_special_tokens=False)) < min_tokens:
                continue
            yield text
        n_yielded += 1
        if n_yielded >= n:
            return


def corpus_tag(specs: list[DomainSpec], n_per: int, min_tokens: int) -> str:
    """A short identity for a whole panel, for use in cache and result paths.

    Every input that changes the documents goes into the hash. This is the
    repo's recurring failure mode: a cached artifact whose path omits one of
    its inputs is served stale, the run succeeds, and the numbers are silently
    from a different corpus. Swapping ``openwebmath`` for another dataset, or
    changing ``n_per``, has to produce a different tag or the digests from the
    previous panel are reused under the same prompt keys.
    """
    payload = ";".join(spec.identity() for spec in specs)
    payload += f";n={n_per};min_tokens={min_tokens}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"dom{len(specs)}x{n_per}_{digest}"


def load_domain_corpora(
    specs: list[DomainSpec],
    *,
    tokenizer,
    n_per: int,
    max_chars: int = 2000,
    min_tokens: int = 128,
    cache_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Materialize the panel to ``{name: [text, ...]}``, cached to one JSON file.

    Cached as a single file for the same reason the wikitext controls are: each
    shard builds the prompt list and strides it, so if two shards disagreed
    about which document is ``openwebmath/5`` their running sums would not be
    poolable -- and nothing would report an error. One file makes agreement
    structural rather than a bet on four HuggingFace streams staying
    deterministic across eight machines.

    Raises:
        ValueError: If any corpus yields fewer than ``n_per`` documents. A short
            group would silently become a smaller-N average, which is exactly
            the comparison the null is meant to control.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if all(len(cached.get(s.name, [])) >= n_per for s in specs):
                return {s.name: cached[s.name][:n_per] for s in specs}

    corpora: dict[str, list[str]] = {}
    for spec in specs:
        logger.info("streaming %s from %s", spec.name, spec.dataset)
        texts = list(
            domain_prompts(
                spec,
                tokenizer=tokenizer,
                n=n_per,
                max_chars=max_chars,
                min_tokens=min_tokens,
            )
        )
        if len(texts) < n_per:
            raise ValueError(
                f"{spec.name}: only {len(texts)} documents of >={min_tokens} "
                f"tokens, wanted {n_per}"
            )
        corpora[spec.name] = texts

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(corpora, indent=1))
    return corpora


#: Replicate draws, registered so they are addressable by name from the CLI.
#:
#: gsm8k and wikitext_a only: those are the two arms the headline rests on
#: (11/39 against 1/39), and they are what an error bar has to be put around
#: before any further corpus is worth fitting.
REPLICATE_DOMAINS: tuple[DomainSpec, ...] = (
    *replicate_specs("gsm8k", N_REPLICATES),
    *replicate_specs("wikitext_a", N_REPLICATES, prefix="wikirep"),
)
DOMAINS_BY_NAME.update({spec.name: spec for spec in REPLICATE_DOMAINS})


#: The expanded panel. Each entry moves one thing against GSM8K.
EXTRA_DOMAINS: tuple[DomainSpec, ...] = (
    # Same genre as GSM8K -- grade-school word problems with worked arithmetic.
    # If GSM8K's advantage is about the genre rather than the dataset, these
    # reproduce it; if only GSM8K works, the result is idiosyncratic to it.
    DomainSpec(
        "svamp",
        "ChilleD/SVAMP",
        None,
        "train",
        text_field="Body",
        join_field="Question",
        extra_fields=("Equation",),
        pack=True,
    ),
    DomainSpec(
        "aqua_rat",
        "deepmind/aqua_rat",
        "raw",
        "train",
        text_field="question",
        join_field="rationale",
        pack=True,
    ),
    # Math, but symbolic and LaTeX-heavy rather than narrated. Holds "topic is
    # mathematics" fixed while removing the natural-language walk through a
    # calculation, which is what the hypothesis says is doing the work.
    DomainSpec(
        "math_algebra",
        "EleutherAI/hendrycks_math",
        "algebra",
        "train",
        text_field="problem",
        join_field="solution",
        pack=True,
    ),
    # The mirror of arith_words: ordered-scale traversal in natural language,
    # no arithmetic anywhere. See ordered_scale_documents.
    DomainSpec(
        "ordered_scale",
        "synthetic:ordered_scale",
        None,
        "train",
        generator="ordered_scale",
    ),
)
DOMAINS_BY_NAME.update({spec.name: spec for spec in EXTRA_DOMAINS})


#: Within-corpus ablations of GSM8K. Anchored by ``gsm8k`` itself, which is the
#: same stream with no transform, so any difference is the transform alone.
ABLATION_DOMAINS: tuple[DomainSpec, ...] = (
    # Question only: problem framing and topic vocabulary, no worked solution.
    DomainSpec(
        "gsm8k_q",
        "openai/gsm8k",
        "main",
        "train",
        text_field="question",
        pack=True,
    ),
    # Solution only: the derivation, without the problem that motivates it.
    DomainSpec(
        "gsm8k_sol",
        "openai/gsm8k",
        "main",
        "train",
        text_field="answer",
        strip_patterns=(r"<<[^>]*>>", r"\n####.*$"),
        pack=True,
    ),
    DomainSpec(
        "gsm8k_shuffled",
        "openai/gsm8k",
        "main",
        "train",
        text_field="question",
        join_field="answer",
        strip_patterns=(r"<<[^>]*>>", r"\n####.*$"),
        transform="shuffle_sentences",
    ),
    DomainSpec(
        "gsm8k_noent",
        "openai/gsm8k",
        "main",
        "train",
        text_field="question",
        join_field="answer",
        strip_patterns=(r"<<[^>]*>>", r"\n####.*$"),
        transform="scramble_entities",
    ),
    DomainSpec(
        "gsm8k_nonum",
        "openai/gsm8k",
        "main",
        "train",
        text_field="question",
        join_field="answer",
        strip_patterns=(r"<<[^>]*>>", r"\n####.*$"),
        transform="strip_numbers",
    ),
)
DOMAINS_BY_NAME.update({spec.name: spec for spec in ABLATION_DOMAINS})


#: Corpora that separate "multi-step worked derivation" from the domain it
#: happens to be about, and from the entity-rebinding it happens to contain.
DERIVATION_DOMAINS: tuple[DomainSpec, ...] = (
    # Long chain-of-thought traces, mixed domains. If reasoning traces match or
    # beat GSM8K, "mathematics" is finished as an explanation and derivation
    # stands on its own. Kept as raw text like ultrachat, so the arm differs in
    # discourse rather than in the presence of chat control tokens.
    DomainSpec(
        "reasoning_traces",
        "open-thoughts/OpenThoughts-114k",
        None,
        "train",
        text_field="conversations",
        flatten_messages=True,
    ),
    # Worked physics solutions: derivation, mostly prose, little arithmetic.
    DomainSpec(
        "physics_worked",
        "camel-ai/physics",
        None,
        "train",
        text_field="message_1",
        join_field="message_2",
        pack=True,
    ),
    # Code. Assignment and reassignment IS rebinding, with no prose derivation
    # at all -- the sharpest available test of the rebinding hypothesis against
    # the derivation one.
    DomainSpec(
        "code_python",
        "code-search-net/code_search_net",
        "python",
        "train",
        text_field="whole_func_string",
        pack=True,
    ),
    # Narrative with entity tracking and no derivation whatsoever: characters
    # persist and change state across sentences. The complement of code.
    DomainSpec("tinystories", "roneneldan/TinyStories", None, "train", pack=True),
)
DOMAINS_BY_NAME.update({spec.name: spec for spec in DERIVATION_DOMAINS})


#: AQuA-RAT replicates. It scored 14/39 against gsm8k's 11/39 on a single draw,
#: which is inside neither corpus's measured spread -- the ranking is not
#: interpretable until this panel is run.
AQUA_REPLICATES: tuple[DomainSpec, ...] = replicate_specs("aqua_rat", N_REPLICATES)
DOMAINS_BY_NAME.update({spec.name: spec for spec in AQUA_REPLICATES})


#: Round two: what in a *solution* carries the effect, given that digits,
#: sentence order and coreference all turned out not to.
NOTATION_DOMAINS: tuple[DomainSpec, ...] = (
    DomainSpec(
        "gsm8k_noeq",
        "openai/gsm8k",
        "main",
        "train",
        text_field="question",
        join_field="answer",
        strip_patterns=(r"<<[^>]*>>", r"\n####.*$"),
        transform="strip_equations",
    ),
    DomainSpec(
        "equations", "synthetic:equations", None, "train", generator="equations"
    ),
)
DOMAINS_BY_NAME.update({spec.name: spec for spec in NOTATION_DOMAINS})
