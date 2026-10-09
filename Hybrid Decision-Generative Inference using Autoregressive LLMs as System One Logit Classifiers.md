# **Sysone-Edge: Hybrid Decision-Generative Inference using Autoregressive LLMs as System One Logit Classifiers**

**Bernard Chiang Zhi Heng**

3 Oct 2026

*Status: protocol and literature paper. No experiments have been run; all quantitative claims about this system are hypotheses to be tested.*

# **Author Note**

Thanks and acknowledgement to bycloudai, sgoedecke and TheoLeeCJ for the inspiration to start on this exploratory project.
I welcome any collaboration on improvements and expansion on ideas discussed in this paper via email at bczh2000@gmail.com.

# **Abstract**

Software that sits around a language model, such as a retrieval-augmented generation (RAG) router, an agent loop or a safety filter, usually needs a choice from a small fixed set rather than free text. Asking an autoregressive model to emit that choice as JSON adds latency and a failure mode: the output may not parse. Informal "System One" projects (Jev, Laya and Von, as described to the author; we have not been able to cite a primary source for them) suggest an alternative. Format the prompt so that the answer is a single choice index, then read the next-token logits of the allowed choices in one forward pass and generate nothing. This gives a probability distribution over the options at the cost of one prefill. We propose Sysone-Edge, an inference harness that toggles one locally hosted small language model (SLM) between this decision mode ("System One") and open-ended generation ("System Two"), with a shared KV-cache prefix. We specify four integrations (RAG routing and chunk filtering, an endpoint-level toggle, agent state transitions and pre-generation guardrails), a reproducible evaluation on Apple Silicon with Banking77 and a prompt-injection dataset, and three measures: latency, calibration and robustness to context length. **No results are reported here.** The contribution of this paper is the design, the pre-specified protocol and the hypotheses, which are stated so that they can fail.

*Keywords:* System One, System Two, LLM, logit scoring, calibration, KV cache, edge inference

# **Introduction**

The title of this project borrows the dual-process vocabulary of Kahneman [1]: a fast, automatic mode and a slow, deliberate one. The mapping onto language models is an analogy, not a claim about cognition. In practice, much of the traffic through an LLM application is not deliberation at all. It is classification: which tool to call, whether a retrieved passage is relevant, whether a request is safe, whether to ask a human. These decisions are normally obtained by letting the model write text and then parsing it.

Our working hypothesis is that, for an edge-deployed SLM, replacing generated-and-parsed decisions with constrained logit scoring over a cached prefix (i) removes parse failures by construction, (ii) lowers orchestration latency, and (iii) yields confidence scores that can drive state-machine logic. Each part is empirically uncertain. Part (i) is true by construction only for the parsing step; it says nothing about whether the chosen label is correct. Part (ii) is plausible, since the decision needs one prefill and no decoding, but its size depends on the baseline and the hardware. Part (iii) holds only if the scores are calibrated, which is known to vary [2, 3, 4]. Earlier drafts of this project stated a latency gain of "over 3x" and the "entire" elimination of hallucination. We withdraw both as unmeasured. They are now hypotheses H1 and H2 in the Methodology.

**Contributions.** (1) A single-harness design that switches between decision and generation modes on one model and one KV cache. (2) Four concrete integration patterns with explicit decision rules. (3) A pre-specified, reproducible evaluation protocol with named datasets, metrics, baselines and stopping rules. (4) A candid account of what is not yet known.

# **Literature Review**

**Why free-text decisions are fragile**
Software that consumes model output needs it to conform to a grammar. Constrained decoding methods enforce this by masking invalid tokens at each step, for example by compiling a regular expression or grammar into a finite-state machine [5]. Benchmarks of structured output show that coverage of real-world JSON schemas differs widely across constraining engines and that the constraint itself can affect quality [6]. Imposing format restrictions can also lower reasoning performance, depending on the task and how strict the format is [7]. We therefore treat "model fails to emit valid JSON" as a real but quantitatively open problem. The draft's earlier statement that 1.5B to 3B models fail in "up to 32%" of categorization tasks had no verifiable source, and we do not repeat it. Measuring the invalid-output rate of our own baseline is part of the protocol (Section "Assessments and Measures").

**Scoring choices from logits**
Reading choice probabilities directly from a model has precedent. Robinson et al. showed that presenting options with symbols (A, B, C, ...) and scoring the symbol token works well for sufficiently capable models on multiple-choice questions [8]. Two caveats apply to our setting. First, the same work finds that this "multiple-choice symbol binding" ability depends on model scale and instruction tuning. Second, models show selection bias toward particular option positions or labels, and this bias can be large for some models [9]. Our design must therefore control for option order and label tokens (see Methodology). Nothing in the literature we have verified shows that a 4-bit 7B model is a good logit classifier on our tasks; that is what we propose to test.

**Calibration and confidence**
A probability is only useful for thresholding if it is calibrated. Guo et al. showed that modern neural networks are often over-confident and popularized Expected Calibration Error (ECE) and temperature scaling as a measure and a fix [2]. For language models, Kadavath et al. found that large models are reasonably calibrated on multiple-choice formats and that calibration improves with scale and few-shot examples [3]. Xiong et al. found that verbalized confidence tends to be over-confident and that consistency-based methods help, but with limits [4]. Instruction tuning and quantization may degrade token-probability calibration. We do not assume either way and measure it, with and without temperature scaling.

**Prefix reuse and efficient serving**
A decision harness is fast only if the shared context is not recomputed. PagedAttention manages the KV cache in blocks to reduce waste and enable sharing across requests [10]. SGLang adds automatic prefix reuse through RadixAttention and an execution model for structured programs [11]. Prompt Cache reuses precomputed attention states for modular prompt segments, and reports lower time-to-first-token on its own benchmarks [12]. These systems target server GPUs. Their ideas carry over to a single-user machine, but the speedups do not carry over automatically. Whether unified memory on Apple Silicon makes a large cached prefix cheap in practice is an empirical question for this work. Our implementation uses MLX [13], an array framework for Apple silicon.

**Applications: RAG, agents and safety**
In RAG [14, 15], routing and filtering are classification problems. Adaptive-RAG trains a small classifier to pick a retrieval strategy by query complexity [16]. Corrective RAG uses a lightweight evaluator to rate retrieved documents and trigger corrective actions [17]. Both show that small decision components are useful in a RAG pipeline; ours differs in reusing the generator itself as the classifier rather than training a separate one. For agents, ReAct interleaves reasoning and tool actions in free text [18], which is the pattern whose output parsing we want to replace at transition points. For safety, Llama Guard casts input-output moderation as classification by a fine-tuned LLM [19], and prompt injection is an established attack on instruction-following models [20]. We use a general instruction model with a label prompt rather than a fine-tuned guard model, so we expect lower accuracy than specialised classifiers and treat that as a trade-off to quantify.

**Long contexts**
Accuracy on information in long inputs can drop, and depends on where in the context it sits [21]. This motivates the context-length experiment below, in which the classification target is placed at varied positions.

**Gap.** The works above cover constrained decoding, option scoring, calibration, prefix caching and classifier-style RAG components separately. We did not find, in the sources we checked, an evaluation that combines them for a single quantized SLM on consumer hardware with a toggle between decision and generation. This is a limited claim: our search was not a systematic review.

# **Methodology**

To simulate a real-world application accessible to the general public, the testing environment relies entirely on consumer hardware and open-weights SLMs.

## **Hypotheses**

* **H1 (reliability).** System One produces zero unparseable decisions by construction, while the JSON-mode baseline produces a non-zero invalid rate on the same prompts.
* **H2 (latency).** Median end-to-end decision latency of System One is lower than that of the JSON-mode baseline. We pre-register a threshold of at least 2x for the headline claim, because the draft's earlier "over 3x" figure had no basis. Any smaller gain is reported as it is.
* **H3 (accuracy).** System One macro-F1 is within 2 points of the JSON-mode baseline on Banking77 and on the injection set.
* **H4 (calibration).** ECE of System One probabilities is at most 0.05 after temperature scaling fitted on a held-out split.
* **H5 (context).** Accuracy decreases by no more than 3 points between 500 and 8,000 tokens of context.

Each threshold is a design target chosen before data is seen. Failing one is an informative result.

## **Hardware**

| Device | Usage | Features | Description |
| :---- | :---- | :---- | :---- |
| Apple MacBook Air | Primary device running all tests. | M4 chip, 24 GB unified memory, 512 GB storage, macOS 26.7.1 | Apple Silicon unified memory lets CPU and GPU address the same RAM, so a large KV cache can sit in memory without a separate GPU memory pool. Whether this gives a measurable advantage is part of what we test. |

Run conditions (fixed for all experiments): AC power connected, low-power mode off, no other foreground applications, a 2-minute idle cool-down between runs, and chip temperature and thermal-pressure state logged, because a fanless laptop throttles under sustained load.

## **Software**

| Software | Usage | Description |
| :---- | :---- | :---- |
| MLX and mlx-lm | Inference engine. | Apple's array framework for Apple Silicon with lazy evaluation [13]. We use it to get the final-position logits directly and to hold the KV cache. Exact package versions are recorded in a lockfile at the time of the runs. |
| Qwen2.5-Coder-7B-Instruct, 4-bit (`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`) | SLM under test. | A 7B instruction-tuned model from the Qwen2.5-Coder family [22], quantized to 4 bits. It is a code-specialised model, so its choice for routing and safety tasks is a convenience of the author's setup, not a claim that it is the best classifier. The repository commit hash will be pinned. |

The earlier statement that this model shows "exceptional baseline calibration compared to other base models in the same weight class" is removed. No such comparison has been made here. As a robustness check we will repeat the primary experiments with one general-purpose 7B to 8B instruction model if time allows, and report it as secondary.

## **Decision procedure**

For a prompt prefix P (system text, shared context) and a question suffix Q with a candidate set of K labels, each label is assigned a single-token symbol. The procedure is:

1. Prefill P once and keep the KV cache.
2. For each Q, extend the cache with Q and a fixed answer cue (for example, "Answer:") and take the logits z at the last position.
3. Restrict z to the K label tokens and apply a softmax over those entries only. This yields p(label | P, Q), normalized over allowed choices.
4. Return argmax and the full distribution. Do not decode further.

Controls for known biases [9]: the label order is shuffled and the K-way probabilities are averaged over R = 3 fixed permutations (the same seeds for every run) in the accuracy and calibration experiments. Single-permutation latency is reported separately. Multi-token labels (such as `highly_relevant`) are mapped to single-character symbols with a legend in the prompt, so that one forward pass is sufficient.

**Baselines.** (B1) JSON-mode generation: the same model, same prompt content, greedy decoding, with the instruction to output a JSON object {"choice": ...}; invalid output counts as an error and is reported as a separate rate. (B2) Grammar-constrained generation of the same JSON, if the toolchain allows it, as in [5]. B2 is optional and will be reported as unavailable if not implemented.

## **Methodology and Implementation Frameworks**

The core mechanism is to cache the shared context of a prompt once and evaluate different question suffixes or candidate token indices against it. In MLX, a suffix batch is processed in one batched forward pass whenever lengths allow; otherwise suffixes are run sequentially from a copied cache. The following four integrations form the basis of the practical application.

1. RAG Pipeline Integration (The Triage and Routing Layer)
   1. Implementation: Introduce the System One model at the routing and evaluation bottlenecks of a RAG pipeline, in the style of classifier components such as [16, 17] but using the generator itself.
   2. Pre-Retrieval Routing: Instead of an LLM generating "Use Vector Search", the System One pass outputs a probability distribution across discrete choices, e.g. `['vector', 'graph', 'sql', 'none']`.
   3. Post-Retrieval Chunk Filtering: Cache the retrieved document chunks as the prefix. Use the System One forward pass to score the relevance of each chunk, e.g. `['highly_relevant', 'partially_relevant', 'irrelevant']`, against the query. The pipeline only forwards chunks exceeding a threshold to the System Two generative step.
2. Inference Harness / Orchestration Plugin
   1. Implementation: Develop a middleware wrapper (similar in spirit to constrained-decoding libraries [5]) that sits on top of a standard serving interface. The prototype targets mlx-lm only. Support for vLLM or Hugging Face `generate` is future work and not part of the evaluation.
   2. Dynamic Toggling: The wrapper intercepts requests. If a request includes a strict `choices` schema, it halts decoding, evaluates the logits of the choice tokens and returns the highest-probability choice with its distribution. If no schema is provided, it defaults to standard generation. A developer therefore uses one endpoint for software-driven decisions and open-ended text.
3. Agentic State Machine Transitions
   1. Implementation: Agent loops in the style of ReAct [18] depend on parsing free-text tool calls and next steps. The System One model is used strictly for state transitions.
   2. Workflow: At the end of every agent step, the model reads the current state and selects the next node, e.g. `['execute_code', 'write_tests', 'ask_human', 'terminate']`. The agent executes automatically if the top probability is at least a threshold τ, and requests human intervention otherwise. The draft's τ = 0.9 is a starting value only. τ is chosen on the validation split (the lowest value meeting a target selective accuracy) and then frozen before test evaluation.
4. Pre-Generation Semantic Guardrails (Safety and Tone)
   1. Implementation: Apply System One to output constraints without the cost of a second model call. Before the main response is generated, the cached context is scored against a safety schema `['safe', 'unsafe']` or a tone schema `['professional', 'casual']`. If p(unsafe) exceeds a threshold chosen on validation data, generation is blocked. Because a false negative here has asymmetric cost, results report recall at fixed false-positive rate, not accuracy alone.

Only integrations 1 (routing) and 4 (binary injection filter) are evaluated quantitatively in this study through the datasets below. Integrations 2 and 3 are specified and prototyped but evaluated only on latency, because no suitable labelled benchmark is committed to here. This scope limit is deliberate.

## **Assessments and Measures**

* **Latency and Throughput Benchmarking.** Measure Time-to-First-Token and total time per decision for B1 and System One, with the shared prefix cached in both. Vary the number of suffix questions per prefix (1, 4, 16, 64) and the prefix length. Protocol: 5 warm-up runs discarded, then at least 30 timed runs per condition, using a monotonic clock and a synchronisation (evaluation forced) call before stopping the timer, since MLX evaluates lazily. Report median, IQR, min, max and mean with standard deviation. Report the cache-prefill cost separately from the per-suffix cost.
* **Calibration Testing (ECE).** For each prediction take the probability of the predicted label as confidence, bin into 15 equal-width bins and report ECE [2] with a 95% bootstrap interval (1,000 resamples). Also report reliability diagrams, the negative log-likelihood and the Brier score, and repeat after temperature scaling fitted on the validation split only. ECE is known to depend on binning, so we report the bin count.
* **Robustness to Context Length.** Evaluate accuracy as the input grows from 500 to 8,000 tokens (500, 1k, 2k, 4k, 8k) by padding the question with unrelated passages, with the question placed at the start, middle and end [21]. The goal is to check that the logit scorer does not degrade on large RAG contexts.
* **Invalid-output rate.** Fraction of B1 outputs that fail to parse or fall outside the label set, with a Wilson 95% interval. This is the direct test for H1.

### ***Evaluation Datasets and Measures***

| Dataset | Purpose | Description | Example |
| :---- | :---- | :---- | :---- |
| Banking77 (`PolyAI/banking77`) [23] | Intent classification | 77 fine-grained banking intents. Used for multi-class routing accuracy and for ECE. Official train/test split; 10% of train is held out for validation (temperature, thresholds, few-shot examples). | A customer query labelled with one of 77 intents. |
| Prompt-injection set (`deepset/prompt-injections`) | Binary classification | A public Hugging Face dataset of injection and benign prompts. Its size is small, so intervals will be wide. Used as a pre-generation triage filter. The attack class is described in [20]. We cannot yet state its exact size or licence terms beyond the repository page, and will record them with the commit hash at run time. | `['injection', 'clean']` |

Banking77 has 77 classes, so a 77-way symbol set cannot be single-character. We therefore use a two-step scheme: either (a) a numeric index of up to two tokens, scored with a second cached step, or (b) a restricted top-K shortlist of 10 labels from a fast retriever. The primary protocol is (a), and it costs two forward passes for this dataset. We will report this cost explicitly rather than hide it.

**Statistics.** Accuracy and macro-F1 are reported with 95% bootstrap intervals. Paired differences between System One and B1 use a paired bootstrap on the same items. All random seeds, prompts, permutations and the test split are fixed in the repository before running. The test split is run once after validation choices are frozen.

**Reproducibility.** The release will contain the prompts, label maps, seeds, a lockfile with MLX and mlx-lm versions, model and dataset commit hashes, the raw per-item outputs (label, probabilities, latency) and the scripts to regenerate every table. Where it is practical, the energy cost is also logged.

**Threats to validity.** Single device and single model; thermal throttling on a fanless laptop; the code-specialised model may under-perform on non-code tasks; Banking77 and the injection set are public and may appear in pre-training data (contamination), which would flatter accuracy and calibration; prompt-injection data is small and may not represent real attacks; the latency comparison depends on how strong an implementation of B1 we write.

# **Results & Discussions**

**No experiments have been run at the time of writing, so this paper reports no results.** We have deliberately not included placeholder numbers, tables of invented statistics or comparison claims. The tables below are the templates that will be filled by the protocol above, and an empty cell means "not measured".

| Measure | B1 (JSON mode) | System One | Notes |
| :---- | :---- | :---- | :---- |
| Banking77 accuracy / macro-F1 | not measured | not measured | 95% bootstrap interval |
| Injection recall at fixed FPR | not measured | not measured | Threshold from validation |
| Invalid-output rate | not measured | 0 by construction | Wilson interval for B1 |
| Median latency per decision (ms) | not measured | not measured | Prefix cached in both |
| ECE (15 bins), raw / scaled | not applicable | not measured | |
| Accuracy at 500 / 2k / 8k tokens | not measured | not measured | |

**What the literature already suggests (not our findings).** Prior work gives reasons for cautious optimism and for specific worries. Large models can be calibrated on choice-style questions [3], and option-symbol scoring works for capable models [8]. But position and label bias [9], over-confidence in neural networks generally [2] and degradation in long contexts [21] all apply to our design. Format restrictions can also hurt accuracy in either direction depending on the task [7], so a faster decision is not automatically a better one.

**Expected comparison with traditional SLM and RAG pipelines.** If H1 and H2 hold, the main benefit would be operational: lower and more predictable latency at each routing point, and no parse-retry loop. If H3 fails, the speed advantage is worth little for the tasks where accuracy matters, and a constrained-decoding baseline [5] would be the better choice. If H4 fails even after temperature scaling, the thresholded behaviours (auto-execute above τ, block above a safety threshold) are unsafe to deploy, and only the argmax label would be usable. We state these implications now so that they are not chosen after the fact.

**Component-level expectations.** Routing and chunk filtering are the most likely to work because they resemble existing classifier components [16, 17]. The guardrail case is the riskiest: a general 7B model scored with a label prompt is unlikely to match purpose-built guard models [19], and a prompt injection is itself an attempt to override the prompt that holds the label instructions. An attacker could target the classifier prompt directly. The system must be treated as one layer of defence and not as a complete one.

# **Conclusion**

**Status of Objectives**
The goal of this work is to specify and justify a hybrid decision-generative inference harness for edge SLMs, and a protocol to test it. The specification and protocol are complete. The central claims, that logit scoring removes parse failures, reduces latency and gives usable confidence, are **not yet tested**, and this paper does not confirm or validate them.

**Summary of Contributions**
We reviewed the literature on constrained decoding, option scoring, calibration, prefix caching and classifier-style RAG components; defined four integration patterns with explicit decision rules; stated five falsifiable hypotheses with pre-registered thresholds; and described a reproducible Apple Silicon evaluation on Banking77 and a public prompt-injection dataset. Two figures in our earlier draft (a "3x" latency gain and a "32%" JSON failure rate) were removed because we could not support them.

**Limitations and Future Work**
The study is limited to one device, one quantized model and two public datasets, and no results exist yet. The immediate next step is to implement the harness, run the protocol once with the test split frozen, and report all outcomes, including negative ones. After that, the planned extensions are other model families and sizes, a grammar-constrained baseline, agent-level end-to-end evaluation of state-machine integrations, a comparison with fine-tuned guard models, and support for server-side engines such as vLLM.

# **References**

1. Kahneman, Daniel (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
2. Guo, Chuan, Pleiss, Geoff, Sun, Yu, and Weinberger, Kilian Q. (2017). On Calibration of Modern Neural Networks. *arXiv preprint* arXiv:1706.04599. https://arxiv.org/abs/1706.04599
3. Kadavath, Saurav, et al. (2022). Language Models (Mostly) Know What They Know. *arXiv preprint* arXiv:2207.05221. https://arxiv.org/abs/2207.05221
4. Xiong, Miao, Hu, Zhiyuan, Lu, Xinyang, Li, Yifei, Fu, Jie, He, Junxian, and Hooi, Bryan (2023). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *arXiv preprint* arXiv:2306.13063. https://arxiv.org/abs/2306.13063
5. Willard, Brandon T. and Louf, Rémi (2023). Efficient Guided Generation for Large Language Models. *arXiv preprint* arXiv:2307.09702. https://arxiv.org/abs/2307.09702
6. Geng, Saibo, Cooper, Hudson, Moskal, Michał, Jenkins, Samuel, Berman, Julian, Ranchin, Nathan, West, Robert, Horvitz, Eric, and Nori, Harsha (2025). JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models. *arXiv preprint* arXiv:2501.10868. https://arxiv.org/abs/2501.10868
7. Tam, Zhi Rui, Wu, Cheng-Kuang, Tsai, Yi-Lin, Lin, Chieh-Yen, Lee, Hung-yi, and Chen, Yun-Nung (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models. *arXiv preprint* arXiv:2408.02442. https://arxiv.org/abs/2408.02442
8. Robinson, Joshua, Rytting, Christopher Michael, and Wingate, David (2022). Leveraging Large Language Models for Multiple Choice Question Answering. *arXiv preprint* arXiv:2210.12353. https://arxiv.org/abs/2210.12353
9. Zheng, Chujie, Zhou, Hao, Meng, Fandong, Zhou, Jie, and Huang, Minlie (2023). Large Language Models Are Not Robust Multiple Choice Selectors. *arXiv preprint* arXiv:2309.03882. https://arxiv.org/abs/2309.03882
10. Kwon, Woosuk, Li, Zhuohan, Zhuang, Siyuan, Sheng, Ying, Zheng, Lianmin, Yu, Cody Hao, Gonzalez, Joseph E., Zhang, Hao, and Stoica, Ion (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *arXiv preprint* arXiv:2309.06180. https://arxiv.org/abs/2309.06180
11. Zheng, Lianmin, et al. (2023). SGLang: Efficient Execution of Structured Language Model Programs. *arXiv preprint* arXiv:2312.07104. https://arxiv.org/abs/2312.07104
12. Gim, In, Chen, Guojun, Lee, Seung-seob, Sarda, Nikhil, Khandelwal, Anurag, and Zhong, Lin (2023). Prompt Cache: Modular Attention Reuse for Low-Latency Inference. *arXiv preprint* arXiv:2311.04934. https://arxiv.org/abs/2311.04934
13. Hannun, Awni, Digani, Jagrit, Katharopoulos, Angelos, and Collobert, Ronan (2023). *MLX: Efficient and flexible machine learning on Apple silicon* [Software]. https://github.com/ml-explore/mlx
14. Lewis, Patrick, et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *arXiv preprint* arXiv:2005.11401. https://arxiv.org/abs/2005.11401
15. Gao, Yunfan, et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv preprint* arXiv:2312.10997. https://arxiv.org/abs/2312.10997
16. Jeong, Soyeong, Baek, Jinheon, Cho, Sukmin, Hwang, Sung Ju, and Park, Jong C. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. *arXiv preprint* arXiv:2403.14403. https://arxiv.org/abs/2403.14403
17. Yan, Shi-Qi, Gu, Jia-Chen, Zhu, Yun, and Ling, Zhen-Hua (2024). Corrective Retrieval Augmented Generation. *arXiv preprint* arXiv:2401.15884. https://arxiv.org/abs/2401.15884
18. Yao, Shunyu, Zhao, Jeffrey, Yu, Dian, Du, Nan, Shafran, Izhak, Narasimhan, Karthik, and Cao, Yuan (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv preprint* arXiv:2210.03629. https://arxiv.org/abs/2210.03629
19. Inan, Hakan, et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. *arXiv preprint* arXiv:2312.06674. https://arxiv.org/abs/2312.06674
20. Pérez, Fábio and Ribeiro, Ian (2022). Ignore Previous Prompt: Attack Techniques For Language Models. *arXiv preprint* arXiv:2211.09527. https://arxiv.org/abs/2211.09527
21. Liu, Nelson F., Lin, Kevin, Hewitt, John, Paranjape, Ashwin, Bevilacqua, Michele, Petroni, Fabio, and Liang, Percy (2023). Lost in the Middle: How Language Models Use Long Contexts. *arXiv preprint* arXiv:2307.03172. https://arxiv.org/abs/2307.03172
22. Hui, Binyuan, et al. (2024). Qwen2.5-Coder Technical Report. *arXiv preprint* arXiv:2409.12186. https://arxiv.org/abs/2409.12186
23. Casanueva, Iñigo, Temčinas, Tadas, Gerz, Daniela, Henderson, Matthew, and Vulić, Ivan (2020). Efficient Intent Detection with Dual Sentence Encoders. *arXiv preprint* arXiv:2003.04807. https://arxiv.org/abs/2003.04807
