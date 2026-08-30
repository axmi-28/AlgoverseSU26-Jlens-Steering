Implementation
Instructions
We will use this space as the central, shared, living info hub for the project implementation. This is your space, and it’s up to you how to organize it. I generally won’t be writing here. One option is to structure it like a fleshing out of the research proposal. At a minimum, when you make/use other files (spreadsheets, Github repos, Google docs, etc.), please include a link and a brief explanation here so everyone knows where to find it if needed.

Datasets
Locating and Editing Factual Associations in GPT
CounterFact dataset: “challenging evaluation dataset for evaluating counterfactual edits in language models. Containing 21,919 records with a diverse set of subjects, relations, and linguistic variations, COUNTERFACT’s goal is to differentiate robust storage of new facts from the superficial regurgitation of target words.” 
Widely cited and used 
Followup with more methodology: MASS-EDITING MEMORY IN A TRANSFORMER

Analysing the Generalisation and Reliability of Steering Vectors
https://github.com/dtch1997/steering-bench
Anthropics MWE dataset anthropics/evals · GitHub
13 datasets in A/B multiple-choice form

https://github.com/anthropics/jacobian-lens
Includes eval datasets for original j-lens paper

SteeringControl: Holistic Evaluation of Alignment Steering in LLMs
“designed to evaluate representation steering methods by testing whether interventions can reliably steer a specific target behavior while minimizing unintended effects on others. Unlike prior work that focuses narrowly on individual alignment objectives, SteeringControl supports comprehensive evaluation across a diverse set of behavioral axes, enabling controlled comparisons and analysis of behavioral entanglement.” 

Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control
Relevant methodology (also works with Jacobians)
https://github.com/trustworthyrobotics/lqr-activation-steering


Models: Qwen3-8B, Gemma-2 2B 
We can use Neuronpedia API (for free) as well for getting mass results for pure J-lens 
Do not provide activation-level access

1. What is the distribution of prompt-local Jacobians Jx around the averaged Jacobian J̄? How does this vary with causal effects of steering?

2. What is the distribution of local pulled-back directions gx = Jxᵀ uy? Are they aligned, clustered, sign-inconsistent, or dominated by rare prompts? How does this vary with causal effects of steering?

