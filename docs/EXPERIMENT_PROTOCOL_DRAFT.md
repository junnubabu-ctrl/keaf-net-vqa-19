# Draft protocol — not a completed preregistration

Data versions, annotation sample size, evaluator commits and final budget configuration remain to freeze before evaluation. Citation IDs refer to the accompanying bibliography.

# 3. Problem statement, questions and hypotheses

Let $x=(I,q)$ denote an image and question, and let $\mathcal{K}_t$ be a fixed knowledge snapshot. Retrieval produces an ordered set $E_k(x)$. The system generates a candidate $a$ from accepted evidence $S\subseteq E_k(x)$ and either releases $a$ or returns an abstention symbol $\bot$. Ground-truth answers and evidence labels are absent from the inference interface.

The goal is to maximize useful coverage subject to a target selective risk and a fixed inference envelope. This differs from maximizing accuracy over the subset that happens to survive an arbitrary filter. An always-abstaining system has no defined selective risk and no useful coverage; it cannot be counted as a successful verifier.

**RQ1 / H1 — reliability under corrupted evidence.** Under matched retrieval, generator and token budgets, the verification policy will reduce selective risk at a common achievable coverage relative to relevance-only filtering. The primary comparison is the absolute paired risk difference at 50% coverage on the predeclared corruption mixture. H1 is supported if the multiplicity-adjusted interval excludes zero in the favorable direction; it is weakened by an interval spanning zero and rejected for that condition if the interval excludes zero in the unfavorable direction. If 50% coverage is unattainable, H1 is not demonstrated; the coverage failure is reported.

**RQ2 / H2 — independence of the release assessment.** Independently assessed evidence support will improve correctness prediction and reduce unsupported releases compared with an answerer’s self-reported support identifiers. Support requires improved held-out Brier score and a favorable unsupported-release comparison. A gain in self-agreement alone does not support H2. Disagreement between correctness and grounding endpoints is retained as an inconclusive or mixed result.

**RQ3 / H3 — necessity of evidence controls.** Removing deduplication or contradiction handling will increase unsupported releases under the corresponding corruption condition. Support requires a favorable paired difference for the relevant ablation after correction for multiple comparisons. A null or reversed difference weakens or rejects the corresponding component claim. Clean-data coverage loss larger than a predeclared two-percentage-point margin rejects the claim that the component is a practically acceptable default.

**RQ4 — transfer.** How do calibration and coverage change on a distinct knowledge-intensive dataset without refitting? Transfer is secondary and descriptive until data licenses, entity disjointness and compatible scoring are confirmed. No universal calibration or out-of-distribution guarantee is asserted.

The 50% operating point, 10% calibration risk target and two-point clean-coverage margin are proposed preregistration choices, not fitted values or observed results. They must be frozen before final evaluation and may be revised only through a dated amendment using pilot data alone.

# 4. Methodology

## 4.1 Evidence representation and retrieval

Each passage stores an identifier, exact text, source URL, source family, snapshot identifier and license. The full corpus and the ordered retrieved evidence are hashed. A source family denotes potentially dependent provenance, such as mirrors of one article; it is not a calibrated trust rating. The final dataset adapter must additionally retain entity identifiers, temporal qualifiers and passage boundaries. Those adapter fields are not yet implemented.

The implemented retriever applies BM25 to the question and a question-conditioned visual description generated from the input image. Its proposed pilot parameters are $k_1=1.5$, $b=0.75$ and $k=10$. This is genuine corpus retrieval, although lexical retrieval is only a feasibility baseline. A frozen FLMR/PreFLMR or entity-aware retrieval condition is required to establish that gains survive stronger retrieval. The generated description is a model observation, not independently verified evidence.

The knowledge corpus must be a versioned, licensed snapshot prepared before evaluation. Live web search is not part of benchmark inference. Passage generation, chunking and index construction must not use held-out answers. The implementation checks required provenance fields but does not establish their truth or licensing validity merely because a string is supplied.

## 4.2 Support, contradiction and redundancy

For a declarative claim $h$, a relation model estimates textual entailment $u(e,h)$, contradiction $c(e,h)$ and neutral relation. These are model scores. They become empirically validated probabilities only after suitable fitting and held-out assessment. Entailment means that the passage supports the claim as written; it does not prove that either refers to the correct visual entity.

Exact normalized duplicates are collapsed and their provenance aliases retained. Semantic paraphrases and correlated source families require further treatment; the current implementation does not count multiple copies as independent confirmations. Two passages are provisionally in conflict when either directional contradiction score exceeds $\tau_c$. The implemented pilot uses $\tau_c=0.8$ and withholds both endpoints of an unresolved conflict. This conservative rule avoids choosing a winner using an unsupported domain-name prior. It can, however, remove useful evidence when statements differ only by time, entity or scope.

The accepted set is therefore

$$S=\operatorname{dedup}(E_k)\setminus\{e_i:\exists e_j,\max[c(e_i,e_j),c(e_j,e_i)]\geq\tau_c\}.$$

This equation describes the implemented heuristic, not a new learned graph reasoner. A conflict graph is an audit representation; no novelty claim is based on graph construction alone. The relation model rejects inputs exceeding its frozen token limit instead of silently truncating away a contradiction.

## 4.3 Set sufficiency and evidence-conditioned answering

An evidence set is sufficient only when it supports all facts necessary to answer the question, including the link from the image to the relevant entity. A high score for one passage does not prove this condition for a multi-hop question. The current implementation reports a maximum textual support score as a diagnostic proxy. A validated set-level multimodal sufficiency module is not yet available and remains a blocker to the full proposed claim.

The candidate answer is generated as

$$a=G_\theta(I,q,S;p,d,s),$$

where $p$ is the prompt, $d$ the decoding configuration and $s$ the seed. The LLM performs both image-dependent description and final answer generation from the accepted evidence. It is not replaced by an evidence vote. Evidence is serialized as untrusted data, and the prompt instructs the model to return a short answer without executing instructions embedded in passages. This instruction is not a proven prompt-injection defense.

Every condition requires fresh generation when evidence changes. Prediction identity includes image bytes, question, exact ordered evidence and provenance, actual model-file fingerprints, claimed checkpoint revision, rendered chat template, decoding settings, processor settings and seed. This identity prevents reuse of an answer from a different evidence set. Audit logs retain the full inputs needed to reconstruct the identity.

## 4.4 Independent grounding and selective release

The implementation uses a separate NLI model to assess the answer against accepted passages. Its present hypothesis serialization is “Question: … Answer: …”, which is an explicit proxy. A validated conversion to declarative claims and an independent image–entity assessment are required before this can be called multimodal grounding. Independence of model parameters reduces direct self-scoring circularity but does not imply independent errors.

The planned correctness predictor takes textual support, contradiction, retrieval and sufficiency features and fits a simple regularized logistic model on a dedicated fit partition. Its target is expected official answer credit, not an invented binary correctness label. A separate calibration partition chooses the threshold. Its probability interpretation must be assessed with proper scores and reliability plots on evaluation data. This fitted model is not implemented or trained in the current delivery.

Let $z(x)$ be the validated hard eligibility gate, $\hat p(x)$ the fitted score and $\gamma$ a frozen release threshold. The release policy is

$$g_\gamma(x)=z(x)\,\mathbf{1}[\hat p(x)\geq\gamma],\qquad
\hat a(x)=\begin{cases}a,&g_\gamma(x)=1,\\\bot,&g_\gamma(x)=0.\end{cases}$$

The policy must abstain when evidence is empty or grounding fails. A high answer-confidence score cannot override a failed gate. The provided threshold utility maximizes empirical calibration coverage among admissible tied-score thresholds. It offers no finite-sample risk guarantee. If no admissible threshold exists, it returns no release point rather than fabricating one.

For official answer credit $v_i\in[0,1]$, coverage and selective risk are

$$C(\gamma)=\frac{1}{n}\sum_i g_\gamma(x_i),\qquad
R(\gamma)=\frac{\sum_i g_\gamma(x_i)(1-v_i)}{\sum_i g_\gamma(x_i)}.$$

Risk is undefined when the denominator is zero. Scores tied at a threshold are released together. A hard-gated policy may not reach full coverage; its maximum coverage must accompany any area summary.

## 4.5 Models, configuration and computational cost

The executable local backend targets Qwen/Qwen2.5-VL-7B-Instruct [@R55], revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, and cross-encoder/nli-deberta-v3-base, revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`. These revisions were obtained from the model host on 8 September 2026; the corresponding weights were not downloaded or run. Both model cards declare Apache-2.0. The NLI card specifies contradiction, entailment and neutral label order. Model licenses do not license the benchmark images or knowledge corpus.

The proposed feasibility environment is Python 3.11, PyTorch 2.6.0, Transformers 4.51.3, Accelerate 1.6.0, Pillow 11.2.1 and SentencePiece 0.2.0. It is a candidate pin set, not an installation-validated lockfile. Pilot decoding is greedy, with at most 64 output tokens per call and image processing between 256 and 1024 visual tokens. The fixed envelope allows one description call and one answer call, at most ten retrieved passages, at most 90 directional pair comparisons and ten answer–passage comparisons. A fixed evidence-token cap and passage chunker must still be implemented and frozen.

The simple BM25 implementation scans corpus term counts; production indexing can reduce retrieval cost. Exact deduplication is linear in passage text length. Pairwise contradiction processing is $O(k^2T_{NLI})$, grounding is $O(kT_{NLI})$, and generation cost depends on image, context and output lengths. These expressions describe operation counts, not measured latency. End-to-end wall time, GPU peak memory, CPU memory, token counts and model calls must be recorded, including verification overhead and model-loading conventions.

# 5. Experimental protocol

## 5.1 Data, partitions and licenses

The primary intended datasets are OK-VQA and A-OKVQA direct-answer. FVQA is a proposed evidence-linked diagnostic; InfoSeek or Encyclopedic VQA is a proposed transfer setting. Dataset choice is predeclared, but actual version, image rights, knowledge snapshot, evaluator commit and exclusion lists are not yet frozen. Therefore no final evaluation may begin under this draft protocol.

Official training data will be divided by image and, where available, entity cluster into development, correctness-fit and threshold-calibration partitions. The proposed ratio is 70:15:15 using seed 17, after cluster reconciliation. Official evaluation data remains untouched. Repeated questions, paraphrases and all corruptions of an original sample stay in the same cluster. If official boundaries share an entity, retain the official primary score and report a separately defined entity-disjoint sensitivity analysis; do not silently alter the benchmark. Public validation labels cannot become a repeatedly tuned test set.

The inference JSON schema contains only question ID, question and image path. Answer labels, evidence labels and split assignments are maintained in separate evaluation manifests. Existing software checks detect intersecting supplied cluster IDs, but constructing correct real-data clusters remains an adapter responsibility. Pretraining contamination cannot be ruled out for the backbone and must be discussed explicitly.

## 5.2 Comparators and component controls

The minimum comparison includes no external knowledge, unfiltered retrieval, relevance-only filtering and the full verified-evidence policy. Each uses identical input questions, knowledge access and answer backbone. Relevance-only and unfiltered conditions receive the same fixed context allowance; verification does not receive more generator calls. Report both a common generator/retrieval envelope and actual total cost, since adding NLI computation is not cost-free.

Strong controls include likelihood-based selective answering, an AvgBLEU-style sampled-answer selector, and a capped ReCoVERR-style evidence-verification condition [@R35; @R33]. Sampling methods require a separate matched-total-generation-budget track. Adaptations must be named as adaptations, with departures from original algorithms disclosed. SURf and ReflectiVA are trained controls: comparing their published numbers against a different Qwen pipeline is not a fair experiment [@R25; @R60]. Either reproduce them with controlled capacity and data, or label the comparison non-equivalent and avoid a superiority claim.

Ablations remove contradiction withholding, duplicate collapse or the independent grounding gate; replace the latter with answerer self-support; and compare raw versus fitted release scores. A declarative-hypothesis assessment is compared with QA serialization. A stronger retrieval condition checks whether a verifier gain survives improvements in the evidence pool. Adaptive search-depth learning is reserved for Paper 5.

## 5.3 Evidence corruption and annotation

Clean retrieval means unmodified retrieved evidence, not guaranteed correct evidence. Natural retrieval errors are evaluated separately from injected corruptions. Controlled conditions include an entity-confusable passage from the same topic, a temporally incompatible claim, removal of necessary support, duplicated text and a meaning-changing contradiction. Additions preserve the context budget by replacing passages rather than expanding it. Proposed replacement severities are 20% and 40% of the retrieved set, with sample-linked seeds 17, 42 and 101. These severities and the equal-weight mixture must be frozen before evaluation.

Corruptions must be plausible, provenance-marked transformations retained only in experiment inputs; they must never modify source records. Two independent annotators, blinded to method outputs and corruption condition where practical, label passage relevance, entity match, factual support, contradiction and set sufficiency. An adjudicator resolves disagreement. Annotators record supporting source spans and the reason that a set is insufficient. Report agreement and uncertainty, not only adjudicated labels. Answerer confidence or improvement after adding a passage cannot substitute for these labels.

The annotation sample is selected before observing model differences and stratified by dataset and corruption family. A pilot estimates disagreement and variance; the final sample size follows a cluster-aware precision or power calculation. No sample-size adequacy is claimed before that calculation. Annotation labor and authorized access to images and source evidence are explicit resource requirements.

## 5.4 Endpoints and statistical analysis

Official answer credit is the primary correctness measure. Use the official benchmark normalizer and evaluator at a recorded commit. The repository’s legacy simplified VQA score is excluded from paper results. Evidence precision and recall use independently adjudicated support labels; report set sufficiency, contradiction detection and unsupported release separately. For unanswerable cases, report correct abstention and unnecessary abstention without treating all refusal as success.

Report coverage at the calibration-selected 10% target risk together with the observed evaluation risk; meeting the calibration target does not guarantee meeting it later. Report common-coverage risk, the full achievable curve and maximum gated coverage. For ungated rankings, a discrete area convention averages cumulative risk at all ranked prefixes with a predeclared tie convention. For gated policies, report area only over achievable coverage, its integration convention and the interval covered; never insert a fictional zero-risk origin or extrapolate to full coverage.

Correctness-score evaluation includes Brier score and a reliability diagram with fixed bins; ECE is secondary and its binning is disclosed. Use paired cluster bootstrap intervals with 10,000 resamples, resampling original image/entity clusters while keeping corruption variants together. Seeds quantify pipeline variability and are not independent substitute samples. Apply Holm correction across the predeclared H1–H3 confirmatory contrasts; treat additional slices as exploratory. Preserve zero-coverage conditions, failed runs, negative effects and sensitivity to source dependence.

Latency is measured with warmup and device synchronization, reporting median and tail latency with batch size and hardware. Record GPU peak allocated memory, CPU memory and tokens. An initially proposed 24–48 GB GPU is a feasibility resource estimate, not a measured requirement. No paid API calls or cloud jobs have been launched.

