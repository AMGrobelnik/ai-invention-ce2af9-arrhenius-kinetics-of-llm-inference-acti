# Softmax-Boltzmann Prior Art Survey & T_thresh Novelty Framing

## Summary

Six canonical references establish the softmax=Boltzmann identity: (1) Bridle (1990, NATO ASI Neurocomputing) introduced the 'normalised exponential' (softmax) for probabilistic NN output — cited directly by Goodfellow et al. (2016) in their softmax section; (2) Hinton, Vinyals & Dean (2015, arXiv:1503.02531) gave the exact temperature-scaled formula q_i = exp(z_i/T)/Σ exp(z_j/T) and coined 'soft targets'; (3) Goodfellow, Bengio & Courville (2016) cover softmax in Section 6.2.2.3 and Boltzmann machines in Chapter 20; (4) Murphy (2012) treats softmax as log-linear model (Ch. 8) and Boltzmann distribution in undirected graphical models (Ch. 19); (5) Wikipedia cites Boltzmann (1868) and Gibbs (1902) as physics primaries in its 'Statistical mechanics' subsection; (6) Luce (1959) derived the softmax form independently from the choice axiom. T_thresh PRIOR ART VERDICT: CLEAR — no prior work derives T_thresh=Ea/ln(N) or equivalent per-instance recovery threshold. The closest work is SMART (arXiv:2506.23492, June 2026) which derives a CALIBRATION bound T ∈ (−g/log((1−p̂)/p̂), −g/log((1−p̂)/(p̂(n−1)))) for achieving a target confidence p̂ — fundamentally different from T_thresh which asks 'at what minimum T does the currently-suppressed correct token become the argmax?' The logit gap in SMART is max-vs-second (regardless of correctness); in T_thresh Ea is the gap between correct and top-wrong token. Renze & Guven (2402.05201) confirmed zero Boltzmann/Arrhenius/activation-energy mentions. Five MCQA citations collected: MedQA (12,723 USMLE questions), JMIR token-probability medical overconfidence study (AUROC 0.71–0.87), MMLU-Pro (12,000+ 10-option questions, NeurIPS 2024), MMLU (57 tasks), LEXam (7,537 law exam questions). Revised novelty paragraph and applied-scope paragraph drafted, ready for paper integration.

## Research Findings

## Phase 1: Canonical Softmax-Boltzmann References (6 confirmed)

**Bridle (1990)** [1] introduced the 'normalised exponential' (softmax) for probabilistic neural network output in the NATO ASI Neurocomputing volume. He does NOT introduce temperature as a free variable. Goodfellow et al. (2016) cite Bridle (1990) directly in their softmax section (Section 6.2.2.3).

**Hinton, Vinyals & Dean (2015)** [2] (arXiv:1503.02531) provide the canonical temperature-scaled softmax formula extracted from the PDF: *q_i = exp(z_i/T) / Σ_j exp(z_j/T)*, where *'T is a temperature that is normally set to 1. Using a higher value for T produces a softer probability distribution over classes.'* They coin the term 'soft targets' and note that gradient magnitudes scale as 1/T².

**Goodfellow, Bengio & Courville (2016)** [3] cover softmax in Section 6.2.2.3 ('Softmax Units for Multinoulli Output Distributions'), citing Bridle (1990). Chapter 20 covers Boltzmann machines as energy-based models with P(x) ∝ exp(−E(x)/T).

**Murphy (2012)** [4] covers softmax as multinomial log-linear model (Chapter 8) and Boltzmann/Gibbs distributions in Chapter 19 (undirected graphical models).

**Boltzmann (1868) & Gibbs (1902)** [5]: Wikipedia's softmax 'Statistical mechanics' subsection states: *'In statistical mechanics, the softargmax function is known as the Boltzmann distribution (or Gibbs distribution)'*, citing these as primary sources.

**Luce (1959)** [6]: *Individual Choice Behavior* derives the softmax form from the independence of irrelevant alternatives axiom, 30 years before ML usage.

## Phase 2: T_thresh Prior-Art Verdict — CLEAR

No prior work derives T_thresh = Ea/ln(N) or any equivalent per-instance threshold for correct-answer recovery [7–9]. The closest work is **SMART (arXiv:2506.23492)** [8] which proves Proposition 3.2 (g-Boundedness): T is bounded by −g/log((1−p̂)/p̂) < T ≤ −g/log((1−p̂)/(p̂(n−1))), where p̂ is a desired calibration target confidence. This serves post-hoc calibration (reaching target confidence p̂), not per-instance recovery (making the correct token the argmax in overconfident-error cases). Key distinction: SMART's logit gap is max-vs-second (irrespective of which is correct); T_thresh Ea is the gap between the correct token and the top wrong token. SMART should appear in related work with explicit contrast, not as prior art. TURN (2502.05234) [9] uses task-level entropy inflection, not per-instance logit-gap formulas. Renze & Guven (2402.05201) [7] confirmed: zero mentions of Boltzmann, Arrhenius, activation energy, statistical mechanics, logit gap, or Gibbs.

## Phase 3: Applied MCQA Citations (5 collected)

**MedQA** (Jin et al. 2020, arXiv:2009.13081) [10]: 12,723 English USMLE questions, free-form multiple-choice. Canonical medical MCQA benchmark.

**JMIR Token Probabilities** (2025) [11]: 2,522 medical licensing MCQA questions; token probabilities outperform verbal confidence (AUROC 0.71–0.87 across 9 LLMs); uses T=0 primary, T=0.5 sensitivity — directly motivates per-instance temperature calibration.

**MMLU-Pro** (Wang et al. 2024, arXiv:2406.01574, NeurIPS 2024 Spotlight) [12]: 12,000+ questions, 10 options (A–J), Health and Law categories included. Canonical 10-option single-token MCQA benchmark.

**MMLU** (Hendrycks et al. 2020, arXiv:2009.03300) [13]: 57 tasks including law and professional ethics; near-random performance on legally important subjects.

**LEXam** (2025, arXiv:2505.12864) [14]: 7,537 questions across 340 law school exams (English and German).

## Phase 4: Renze & Guven Boltzmann Check

Confirmed absent via PDF fetch_grep: no Boltzmann, Arrhenius, activation energy, statistical mechanics, logit gap, or Gibbs in arXiv:2402.05201.

## Sources

[1] [Bridle (1990) — Probabilistic Interpretation of Feedforward Network Outputs (Semantic Scholar)](https://www.semanticscholar.org/paper/Probabilistic-Interpretation-of-Feedforward-Network-Bridle/1f462943c8d0af69c12a09058251848324135e5a) — Canonical ML introduction of the normalised exponential (softmax). NATO ASI Neurocomputing vol. 68. Does not introduce temperature. Cited by Goodfellow (2016) in softmax section.

[2] [Hinton, Vinyals & Dean (2015) — Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531) — Exact formula extracted: q_i = exp(z_i/T)/sum_j exp(z_j/T). Introduces temperature T explicitly; coins 'soft targets'. Canonical ML reference for temperature-scaled softmax.

[3] [Goodfellow, Bengio & Courville (2016) — Deep Learning (MIT Press)](https://mcube.lab.nycu.edu.tw/~cfung/docs/books/goodfellow2016deep_learning.pdf) — Section 6.2.2.3 introduces softmax for multinoulli outputs, citing Bridle (1990). Chapter 20 covers Boltzmann machines as energy-based models.

[4] [Murphy (2012) — Machine Learning: A Probabilistic Perspective (MIT Press)](https://research.google.com/pubs/archive/38136.pdf) — Chapter 8: softmax as multinomial log-linear model. Chapter 19: Boltzmann/Gibbs distributions in undirected graphical models. Table of contents confirmed.

[5] [Wikipedia: Softmax function — Statistical mechanics subsection](https://en.wikipedia.org/wiki/Softmax_function) — Exact quote: 'In statistical mechanics, the softargmax function is known as the Boltzmann distribution (or Gibbs distribution)'. Cites Boltzmann (1868) and Gibbs (1902) as primary sources.

[6] [Luce (1959) — Individual Choice Behavior: A Theoretical Analysis (via Wikipedia)](https://en.wikipedia.org/wiki/Luce%27s_choice_axiom) — Choice-theoretic derivation of softmax from IIA axiom (1959). Predates ML usage by 30 years. 'The matching law selection rule is sometimes called the softmax function, or the Boltzmann distribution.'

[7] [Renze & Guven (2024) — The Effect of Sampling Temperature on Problem Solving in LLMs](https://arxiv.org/pdf/2402.05201) — Confirmed ABSENT: zero mentions of Boltzmann, Arrhenius, activation energy, statistical mechanics, logit gap, Gibbs. Uses T grid 0.0–1.0 step 0.1, no significant aggregate effect found.

[8] [SMART: Sample Margin-Aware Recalibration of Temperature (2026)](https://arxiv.org/pdf/2506.23492) — RELATED but DISTINCT work. Proposition 3.2: T bounded by -g/log((1-p̂)/p̂) to -g/log((1-p̂)/(p̂(n-1))). Purpose: calibration to target confidence p̂. Different from T_thresh which finds recovery threshold for overconfident errors.

[9] [TURN — Optimizing Temperature for LLMs with Multi-Sample Inference (2025)](https://arxiv.org/html/2502.05234v2) — Task-level entropy inflection point method. No per-instance logit-gap formula. Single temperature per model-task pair, not per-instance.

[10] [Jin et al. (2020) — MedQA: What Disease does this Patient Have?](https://arxiv.org/abs/2009.13081) — 12,723 English USMLE questions, free-form multiple-choice. Canonical medical MCQA benchmark. Submitted to AAAI 2021.

[11] [JMIR (2025) — Token Probabilities to Mitigate LLM Overconfidence in Medical Questions](https://www.jmir.org/2025/1/e64348) — 2,522 medical licensing MCQA questions; token probabilities AUROC 0.71–0.87 outperform verbal confidence across 9 LLMs. Uses T=0 primary, T=0.5 sensitivity. Direct motivation for per-instance temperature calibration.

[12] [Wang et al. (2024) — MMLU-Pro (NeurIPS 2024 Spotlight)](https://arxiv.org/abs/2406.01574) — 12,000+ questions, 10 options (A-J), 14 domains including Health and Law. Canonical 10-option single-token MCQA benchmark. Reduces chance baseline from 25% to 10%.

[13] [Hendrycks et al. (2020) — Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) — 57 tasks including US law and professional ethics. Establishes single-letter answer MCQA as standard format. Near-random LLM performance on legally important subjects.

[14] [LEXam (2025) — Benchmarking Legal Reasoning on 340 Law Exams](https://arxiv.org/abs/2505.12864) — 7,537 questions across 340 law school exams in English and German. Mixed MCQA and open-ended format. LLM-as-Judge evaluation with human expert validation.

[15] [Phase Transitions in the Output Distribution of Large Language Models (2024)](https://arxiv.org/abs/2405.17088) — Studies distributional changes as a function of temperature. No per-instance threshold formula derived. Not a novelty conflict.

## Follow-up Questions

- Does Luce (1959) choice axiom derivation supersede Bridle (1990) as the canonical softmax antecedent, and should both be cited to establish the identity's pre-ML history in the Related Work section?
- Does the SMART paper (arXiv:2506.23492) Proposition 3.2 g-Boundedness warrant explicit comparison in the paper's related work section, distinguishing calibration-to-target-confidence from per-instance recovery threshold T_thresh=Ea/ln(N)?
- Are there any papers post-mid-2026 applying Boltzmann/Arrhenius analogy to LLM inference, given this is a rapidly active area that may have progressed since the search date?

---
*Generated by AI Inventor Pipeline*
