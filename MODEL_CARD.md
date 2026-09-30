# Model card: J-lens position-pair steering

## Artifact and base model

This is an inference-time research intervention and evaluation toolkit, not a
new pretrained or fine-tuned language model. The primary base model is
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), a decoder language model with
36 layers and residual width 4096. The release pins its weights and tokenizer
to `b968826d9c46dd6066d109eabc6255188de91218`, the revision found in the local
cache at release preparation. Historical remote runs did not record the Hub
revision, so this pin is not proof of the revision used for every original run.

Runs use raw text completions, not a chat template or an explicit reasoning
mode. The standard dtype is bfloat16; finite-difference validation uses
float32. The seven edited layers are 13, 16, 19, 22, 25, 28 and 31. Additional
model configurations are exploratory and do not establish replication of the
paper's findings across model families or scales.

## Construction and inputs

The toolkit forms token directions `v_y = J^T u_y`, where `u_y` is a raw
unembedding row. It separates same-position (`D`) from strictly later-position
(`O`) derivatives under shared normalization, and evaluates
`J_lambda = J_O + lambda * J_D`. These labels index token positions, not matrix
diagonals in hidden coordinates.

The principal stored component digest contains 70 token directions for full,
diagonal and off-diagonal lenses fitted on GSM8K and WikiText: 32 documents per
corpus, 128-token windows, initial 16 positions skipped. Its target-position
sampling stride is **4**. The decomposition is exact on those sampled targets;
it is an estimate of the all-target-position average. The public reference
lens from [Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens) is a
separate artifact and has different fit counts. Do not confuse it with these
32-document lenses.

Corpus comparisons include GSM8K question/solution text, WikiText, AQuA-RAT,
SVAMP, competition algebra and OpenWebMath. Source selection, packing and
transformations are in `src/jsteer/domains.py`. The component-fit text snapshot
and fitted directions are included; original dataset revisions and the exact
text snapshots for every historical replicate were not recorded. Third-party
model/data terms remain applicable; see [notices](THIRD_PARTY_NOTICES.md).

## Intervention and evaluation

A rank-2 coordinate swap exchanges source and target coordinates in the
selected residual stream. The concept-panel unit-column variants compute
edits on clean activations and match each layer/position's edit norm to the
full-lens reference. Whole-prompt and prefix-only interventions are distinct.
Generation proceeds after the original prompt without further intervention.

| Evaluation | Population and measurement |
|---|---|
| Flexible generalization | 192 constructed trials; 144 clean-correct, including 39 numerical and 105 non-numerical trials |
| Latent intermediates | 55 clean-correct, covered items from a 90-item probe-swap set; includes 8 pilot items |
| Concept use | 33 entities, two latent clues each; 661 eligible units and 1,339 relational queries in the complete interpolation runs |
| Collateral effects | Next-token KL on neutral queries |

Answer success, explicit naming and target-name leakage are scored separately.
A correct target name on a naming query does not demonstrate correct relational
use. Concept outcome rates average relational queries within each unit before
averaging units; margins/gradient reports operate on relational queries.

At strength 1 on the latent panel, full-to-off-diagonal filtering changes
GSM8K answer hits from 15/55 to 25/55 and leakage from 21/55 to 5/55;
WikiText changes from 13/55 to 23/55 and 21/55 to 9/55. These are reference
results to recompute, not guarantees for new models or prompts.

## Intended use and limitations

Intended for interpretability research and replication of intervention
experiments. It is not a production model-control method, a factuality fix, or
evidence that a particular direction uniquely represents a semantic concept.

The main evidence is on one model. Development panels were reused, and some
strengths were selected on the evaluated subset. The concept panel is held out
from lens fitting, but its later follow-ups are not independent confirmations.
Results depend on edit placement, strength, grading and normalization. Filtering
can reverse its advantage under other placements/doses and can perturb neutral
predictions. Bootstrap intervals condition on fixed lenses and do not capture
all model/fitting uncertainty. See [the paper audit](docs/PAPER_AUDIT.md) for
unresolved discrepancies and [the protocol](docs/PROTOCOL.md) for analysis choices.

## Compute and reproducibility

Reference analysis runs on CPU without base-model weights. GPU experiments
were launched with an A100-80GB default; memory depends on cotangent batching,
sequence length and float32 validation. No total GPU-hour measurement was
recovered. The tested local package snapshot is provided, but the original
remote image used version ranges; bitwise GPU reproduction is not established.
The release does not contain base-model weights or claim a new model license.
