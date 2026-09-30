# Evaluation and analysis protocol

## Populations

Flexible generalization has 64 base prompts and 192 source/target trials. Report
success on the clean-correct subset, using generated-answer regrading in
`25_domain_causal.py`; first-token-only scores have numeric/capitalization edge
cases. The square task contains truncated answer keys in the upstream set and
is a known limitation. The original 105 non-numerical trials and the 105
ordered-scale trials are distinct evaluations.

The latent panel filters upstream probe-swap items by clean correctness,
counterfactual incorrectness and coverage in the 70-token component bank.
The eight pilot items are identified in `data/protocols/` and remain part of the
55-item development expansion. The paired report gives both all-item and
pilot-excluded results. The frozen JSON plan records original source hashes,
input hashes, item ordering, strengths, arms and contrasts; do not regenerate
it just to make old outputs appear current.

The concept panel contains 33 entities with two latent clues each and explicit
name controls. Source/target answers must be disjoint; only queries correctly
answered by the clean model enter eligibility. Eligible units pair a source
entity/clue with a target entity of the same type. Naming, relational use and
neutral queries are separate. Frozen eligibility precedes intervention scoring.

## Edits

The main multi-layer concept edit is computed from clean activations and added
as a fixed tensor at each selected layer/position. Unit-column (`_cn`) variants
remove source/target norm-ratio differences, then match every edit-row norm to
the full-lens reference. Native-column (`_m`) variants retain the ratio. Frozen
writer controls replace only the target writer while preserving full-lens read
coefficients; matching their target term does not match their total edit norm.

Same-position weight varies over 0, .125, .25, .5, .75, 1, 1.5 and 2. Whole-prompt
and prefix-only results are separate conditions. Generation is greedy and the
intervention applies only to original prompt positions. Name leakage and answer
success can overlap under substring scoring.

## Follow-ups and uncertainty

Outcome means weight units equally after within-unit query averaging. Paired
bootstrap draws keep conditions together, clustering by source entity on the
concept panel and by supplied category on the latent panel. Intervals are
pointwise, conditional on fitted directions, with no multiplicity correction.
Corpus-fit replicates characterize fitting variation separately.

Magnitude analysis models within-trial margin changes with lambda indicators
and a five-knot restricted cubic spline of log natural edit size. Natural clean,
arriving and relative sizes are distinct from the applied row-matched norm.
Poor overlap makes relative-size adjustment extrapolative. Dynamic runtime
Householder reflection partly cancels earlier edits in a multi-layer stack, so
it is not interpreted as a valid isolated size control.

Gradient analysis uses clean-state total derivatives of relational and leakage
sequence-log-probability margins, dotted with the actual applied edits and
summed across sites. The original bf16 epsilon check was quantization-limited;
its replacement is a float32 central-difference check on a fixed hash-ordered
subsample, at epsilons .01, .03 and .1. Primary within-swap comparisons subtract
lambda=0. Source/target clustering and single-token subsets are sensitivities.
The path-integration follow-up was not triggered by the specified correlation/
sign thresholds. See saved reports for outcomes rather than treating these
post-development follow-ups as independent preregistered confirmations.
