# Third-party sources and licenses

Original project code is licensed under the MIT License in `LICENSE`.
Third-party model weights, dataset text and dependency code retain their own terms.

- **J-lens code and upstream evaluation JSON:** Anthropic's
  [jacobian-lens repository](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e).
  The dependency and included evaluation files use that exact commit. Its
  [license](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/LICENSE)
  is retained in `docs/licenses/jlens.txt`.
- **Qwen3-8B:** [Qwen model card](https://huggingface.co/Qwen/Qwen3-8B) and
  [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/LICENSE).
  Base-model weights are downloaded separately, not redistributed in this repo.
- **Published reference lenses:**
  [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens).
  Downloaded separately. Fitted pullback summaries in this release were computed
  by the experiment code and are distinct from the published reference lens.
- **GSM8K:** [OpenAI dataset card](https://huggingface.co/datasets/openai/gsm8k).
  Question/solution text is used for fitting; the component-fit cache retains
  original sampled text.
- **WikiText:** [Salesforce dataset card](https://huggingface.co/datasets/Salesforce/wikitext).
  Wikipedia-derived fitting text retains its upstream attribution and terms.
- **Auxiliary fitting corpora:** source identifiers for the preserved cache
  groups are listed below. Transformations/packing are documented in
  `src/jsteer/domains.py`; this repository does not assert ownership of that text
  or replace the individual datasets' terms.

Python packages retain their respective licenses. Optional Gemma configurations
require the upstream model's access conditions and license acceptance; they are
not required to reanalyze the primary paper results.

| Cached group | Upstream dataset |
|---|---|
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| `wikitext_a` | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) |
| `openwebmath` | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) |
| `arith_words` | [synthetic:arithmetic_words](https://huggingface.co/datasets/synthetic:arithmetic_words) |
| `gsm8k_q` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| `gsm8k_sol` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| `aqua_rat` | [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat) |
| `math_algebra` | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| `svamp` | [ChilleD/SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) |
| `ordered_scale` | [synthetic:ordered_scale](https://huggingface.co/datasets/synthetic:ordered_scale) |
| `equations` | [synthetic:equations](https://huggingface.co/datasets/synthetic:equations) |
| `reasoning_traces` | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| `ultrachat` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
