 Investigating Steering with J-lens
Instructions
Proposals must be specific and detailed so the direction of the research is clear. If you’re not sure about a given item, please mark it as such and we can discuss.
Add text using your assigned color for all tabs in this doc:
Melwina, Ankith, Andy, Ishan, Claire
Relevant Past Papers
How is it done today, and what are the limits of current practice?

Things to include for each paper:
One sentence summary of the paper
Gap / limitation of the paper

Recent j-lens related papers
Verbalizable Representations Form a Global Workspace in Language Models
Introduces the Jacobian lens: a refinement of the logit lens that reads which concepts an intermediate activation is poised to verbalize. Argues the resulting J-space can be represented as a global workspace, with properties of broadcasting concepts, being able to be causally manipulated, and redirecting certain flexible reasoning
Limitation: 
verbalizations are restricted to single vocabulary tokens, and multi-token concepts are thus neglected.
More centrally for this proposal, the paper provides strong empirical evidence that
 J-Lens directions are causally usable yet lacks a concrete analysis of why the readout should be a valid write direction. The reported failure in J-lens causal interventions is given candidate explanations: concepts that are outside J-space concepts, or simply concepts that do not match a single token 
this leaves a third possibility: that the averaged pullback is simply the wrong direction for this prompt (the averaged and prompt-local Jacobian directions are also not compared) 
Towards surfacing model algorithms with meta-tokens in the J-Space
With Jlens on Qwen3.6 27B, this post identifies “meta-tokens” - J-space tokens revealing non-obvious algorithms in the model
Ie a chinese token for “what does this mean” that causally mediates understanding the context of the prompt better
Or “most likely” token that causally mediates whether the model hedges or not
R-lens: Making J-lens More Faithful on Early Layers — LessWrong
A refinement of J-lens to better measure early layer J-space, surfacing concepts many layers earlier, as well as with directions more causally responsible for eventual answer under ablation
Replaces J-lens's backward-pass gradient with an “LRP-derived relevance coefficient.” Shows early-layer
 "trash tokens" are substantially a measurement artifact and that R-lens directions are more causally responsible for the eventual answer under ablation.
demonstrates that changing the backward rule
 changes causal potency, which is evidence that the choice of pullback operator is a tunable and consequential parameter
They focus on fidelity of the readout (the only causal validation is ablation) 
Background in steering

Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models
Attempts to test whether localizing where a fact is stored enables you to “edit” the fact
(they fail)
Relevance: that “read” methods that interpret where a representation is does not necessarily where interventions make sense
Ie, j-lens essentially structurally does something similar (except it appears to empirically work oftentimes?)

Analysing the Generalisation and Reliability of Steering Vectors
[the title], stress-tests whether steering vectors are reliable in distribution and generalize across many concepts and models (via a released benchmark as well)
In distribution, steerability is highly variable across inputs (rather than a single property of the vector)
Some concepts, random features/context contribute substantially to how well steering works
Sometimes steering produces opposite of intended effect
OOD: steering often generalizes, but in certain cases it is extremely prone to failure based on rephrasing of prompt

Inverted Detection and Control in Steering Vectors
Identifies “inverted steering vectors” which are directions that strongly discriminate for a concept (positive examples align with them more than negative ones) but which counterintuitively suppresses that concept when added, and promotes the concept when subtracted
Across three models, in attention head output spaces however
Relevance: read directions and write directions might counterintuitively correspond? (this might be more a property of attention?)

FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers
Argues that activation space is not Euclidean for steering purposes - rather, the geometry should be the Fisher information metric of the softmax output layer, pulled back through the Jacobian of the remaining layers
Makes a better covector/vector distinction: a probe direction pulled back through the jacobian is a covector (not for writing) whereas converting it into a tangent vector (to be added for steering) requires further steps
Formalizes existing steering methods as each additive updates under different implicit metrics
Relevance: understates the premise of this project - that a direction obtained by pulling a readout back through the Jacobian is not by itself the direction to write along
Limitation: 
J-lens directions may be fundamentally different than ie the covector/vectors they consider here on trained linear probes 
Ie jacobians are (in a way, essentially) optimized to represent intermediate activations and concepts? Different than optimized to predict output for example? 
For this paper, the Jacobians are prompt local (rather than the averaged Jacobian) 

The premise is that the read direction does not automatically transfer to write direction
This has been investigated in other (relevant?) scenarios in the past…. 

[2604.02608] Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens - “Steerable but not decodable.”
These researchers set out to test out similar reading tool called logit lens and a similar writing tool called function vectors, which is built from showing the model a few examples and averaging the internal difference.  They compared 4032 separate test cases spanning 12 different tasks, 6 different AI models from 3 different families and 8 different prompt phrasings for each task.
They expected the reading tool to see nothing and for the pushing to fail, but that failure pattern didn’t really show up. Instead, in every task and model, the pushing (adding the function vectors to the activation space) worked much better than the reading
With the reading tool, they realized that even if the correct answer wasn’t being seen in the activation space, steering it with the function vectors still pushed it to a right answer
They also found that a successful push vector wasn’t the “correct answer” in vector form, rather it was more like an instruction telling the model to go compute the anwer.
They also found that the push worked best when applied early in the network, while the reading tool only started seeing anything in the late layers

Curveball Steering: The Right Direction To Steer Isn't Always Linear (Raval et al., 2026)

The Linear Representation Hypothesis and the Geometry of Large Language Models
Formalises what it means for a concept to be represented linearly. A concept has 
An unembedding representation: a covector, what a probe or a lens might return - roughly representing degree of presence
An embedding representation, a vector in activation space, what is added for the intervention. 
Converting a covector into the corresponding vector requires “whitening by the covariance of the representation space” or “the causal inner product.” Under that metric, concepts that do not causally interfere become orthogonal.

Limitation:
Potentially offers a framework upon which to principally consider Jacobian-lens “reading” then “writing”
Inference-Time Intervention: Eliciting Truthful Answers from a Language Model
Finds truthfulness related attention heads with linear probes and shifts activations along a direction at inference time to improve truthful output. 
They compare two directions derived from the probing setup
 the probe-weight normal, i.e. the discriminative hyperplane, 
the mass-mean shift, i.e. the difference between the truthful and false class means 

They find the mass-mean shift decently better as a steering direction
They note that the directions the model uses to generate true or false statements are very different from the directions found by probing.

On the Non-Identifiability of Steering Vectors in Large Language Models


Motivation	
Things to include:
What limitation or problem are you solving and how do you know it exists
Why is this limitation important
Why does your idea solve it
Why would your idea probably work

When are J-Lens read directions valid write directions?

J-Lens reads an intermediate activation using an averaged downstream Jacobian:

lens(hℓ) = softmax(WU norm(J̄ℓ hℓ)).

For output token y, its J-Lens direction is approximately

vy = J̄ℓᵀ uy,

where uy is the token’s unembedding direction. The original paper writes with this direction either by adding it,

hℓ ← hℓ + αvy,

or by exchanging the coordinates associated with two J-Lens vectors.

Reading and writing are not the same mathematical problem. The readout asks how a candidate activation change would affect an output score. Steering asks the inverse question: which intermediate change will produce a desired downstream change?

If the desired final-layer change is uy, the inverse problem is

Jx δh ≈ uy.

Adding J̄ᵀuy does not solve this equation. The transpose provides the local gradient of the output objective, not an inverse of the downstream computation. Using it as a steering direction introduces assumptions beyond those needed for readout:

1. The current prompt’s Jacobian Jx is sufficiently similar to the averaged Jacobian J̄.
2. The first-order approximation remains accurate at useful steering strengths.
3. Treating a readout covector as an activation vector produces an appropriate write direction.
4. Increasing a token direction produces the intended concept rather than a narrow lexical effect.
5. The perturbed activation remains meaningful to downstream layers.
6. The direction changes the target without uncontrolled effects on other outputs.

There is also a separate composition question. For one prompt,

Jℓ→L(x) = Jm→L(x) Jℓ→m(x),

but generally

E[BA] ≠ E[B]E[A].

The original J-Lens directly averages the end-to-end Jacobian, so basic single-layer use does not assume that averaging commutes with layer composition. This issue arises if we compose separately averaged maps or interpret the averaged directions as a common coordinate system across layers.

Understanding these assumptions matters because J-Lens interventions are used as evidence that the readout identifies representations the model actually uses. An intervention can fail because the readout is wrong, because the averaged direction is inappropriate for the current prompt, or because the read direction is not a good write direction. These explanations should not be conflated.

Anthropic found that lens-coordinate swaps succeeded on 76 of 192 trials at ordinary strength and 101 of 192 at double strength. Failures were concentrated where the source concept was weakly active before intervention. They suggest that some concepts may be represented outside the J-space or may not correspond to a single-token J-Lens vector. The paper does not compare averaged directions with prompt-local Jacobian directions or characterize the distribution of those directions.

Key Ideas/Contributions/Novelty
What are we doing that hasn’t been done before? Things to consider:
What is the novelty of the idea with respect to the existing literature? What is “new”?
What is the main contribution (to science) of this paper? Are we creating an improved method or a benchmark? What new insights or findings are we providing? What research question are we answering that was not answered? 

Contribution is to develop a better understanding of jacobian-space interventions: 
Roughly we hypothesize interventions might have the following potential faults: the averaged direction is unrepresentative of the prompt, or the read direction is intrinsically a poor write direction. The deliverable is to provide an account of when J-lens directions support reliable interventions, and an ex-ante signal for whether a given intervention will work

Methods
How does your idea work? Describe the way you will get your results from the initial step. Make a diagram. 

steering benchmark related papers (maybe Axbench) 

Experimental Setup

How are you going to test your idea to prove that it works?
Think about social science experiments where researchers have a plan of what to make participants do, what to ask them and how to calculate results based on the responses.
What are your baselines for comparison? i.e control group
What models? What datasets? What metrics (if not accuracy)?

What additional analysis do you plan on doing?
How are you going to prove that any improvements in the system are because of your idea alone, i.e. reduce confounding factors and do ablation testing
How does your idea impact the system? What metrics are you quantifying with? 
Visualizations? Statistics?

Setup
Qwen3.6-27B (?) with J-lens fitted from Neuronpedia 
First set of scenarios might be two-hop factual recall prompts? (perhaps from here https://github.com/google-deepmind/latent-multi-hop-reasoning) 

Objectives
Finding distribution of prompt-local Jacobians around the average: 
Collect several hundred prompts per bridge entity spanning all template families and compute the local Jacobian at each, then characterize the spread around the corpus average 
overall deviation, whether it is structured by template versus by entity, and whether the average is dominated by a small number of atypical prompts.

Finding distribution of local pulled-back directions

When are local directions more causally effective than averaged ones?
Steer with both….

Transferability of local directions across prompts. 


Datasets and Evaluation
Which datasets are you going to use to evaluate your method? Or, are you creating your own?
Reference relevant previous papers here.
If you’re training, what dataset will you use for that?
What is the evaluation metric(s)? 
Some tasks are straightforward to measure (e.g accuracy, for mathematical reasoning). Some are much harder (e.g LLM persuasive ability - think about how you would do this).
Benchmarks/Evaluation Sets
Evaluation and comparison with existing systems.
What are your baselines? 
Ideal Results
What’s the best case scenario/what do you hope to demonstrate?
What is your hypothesis? What results would prove it to be true?

Solving the problem means determining when J-Lens directions support reliable interventions and why they sometimes fail. We will know we have made progress if we can answer at least some of the following:

1. What is the distribution of prompt-local Jacobians Jx around the averaged Jacobian J̄? How does this vary with causal effects of steering?
2. What is the distribution of local pulled-back directions gx = Jxᵀ uy? Are they aligned, clustered, sign-inconsistent, or dominated by rare prompts? How does this vary with causal effects of steering?
3. When do local directions outperform averaged J-Lens directions after matching target effect or output-distribution change?
4. Do local directions transfer between prompts, or are they specific to the prompt from which they were calculated?
5. Do properties such as cos(gx, ḡ), local-gradient norm, layer and initial J-Lens activation predict intervention success?
6. Can an ordinary steering direction substantially change behavior while producing little change under the J-Lens readout?
7. Does a high J-Lens readout imply that the corresponding direction is steerable, or can reading and writing dissociate?
8. At what steering strength does the Jacobian’s predicted output change cease to match the actual change?

This falls under causal interpretability, activation steering and the validation of interpretability tools.

Potential Limitations
Computation limits, generalization limits, dataset limits, ethical limits?
