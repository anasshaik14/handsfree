### 5.8 Comparison with existing methods

Table 13 places the four proposed networks against the baselines they were compared with, and lists the principal published methods on each corpus for context without quoting their figures. That omission is deliberate and is the subject of this paper: figures drawn from the literature were produced under heterogeneous protocols, several of which do not state a grouping key and none of which audits the comparison against its own baseline for exposure asymmetry. Placing such a figure in the same column as ours would assert a comparability that Section 5.1 shows does not hold. The column "protocol" therefore carries as much information as the column "result".

**Table 13.** Comparison against the baselines, with published methods listed for context. LOPO = leave-one-person-out; GD = single group-disjoint split; ED = exposure-disjoint comparison stratum (this work). **Every numeric cell is a measurement made in this study** on the aligned per-sample arrays of Section 4.3; rows whose published figures were not reproduced here are marked *not re-measured* rather than filled with a quoted value, because a figure produced under a different protocol is not the same quantity, and tabulating it beside ours would invite exactly the comparison this paper argues against.

| Modality | Corpus | Method class | Protocol | Reported result | Params | Cross-model audit |
|---|---|---|---|---|---|---|
| Gaze | MPIIGaze | Appearance-based CNN [@mpiigaze] | LOPO (as published) | *not re-measured here* | — | ✘ |
| Gaze | MPIIGaze | Pre-activation ResNet [@mpiigaze_pami][@preact] | LOPO (as published) | *not re-measured here* | 77,533 | ✘ |
| Gaze | MPIIGaze | Pre-activation ResNet [@mpiigaze_pami][@preact] | **ED stratum, measured here** | **3.621° angular** (lower bound, EDCP step 5) | 77,533 | **✔ (GERA)** |
| Gaze | MPIIGaze | **Proposed CNN** | **GD + ED, measured here** | **6.439° angular** | 551,810 | **✔ (GERA)** |
| Head pose | 300W-LP → AFLW2000 | Euler-angle CNN [@ruiz] | Corpus-disjoint (as published) | *not re-measured here* | — | ✘ |
| Head pose | 300W-LP → AFLW2000 | 6-D rotation network [@sixdrep] | **Corpus-disjoint, measured here** | **3.482° MAE** | 39,323,398 | **✔ (negative control)** |
| Head pose | 300W-LP → AFLW2000 | **Proposed CNN** | **Corpus-disjoint + ED, measured here** | **7.367° MAE** | 1,142,339 | **✔ (negative control)** |
| Blink | MRL Eye | Geometric EAR rule [@ear] | n/a (training-free) | not scorable on cropped eyes | 0 | ✘ |
| Blink | MRL Eye | Proposed CNN, *random row split* | Random row, measured here | 98.17 % accuracy | 136,066 | n/a |
| Blink | MRL Eye | **Proposed CNN** | **GD (subject), measured here** | **93.67 % accuracy** | 136,066 | n/a (no baseline) |
| Voice | Speech Commands v2 | Production ASR engine [@vosk][@kaldi] | **Official split, measured here** | **95.50 % acc. / 97.43 % macro-F₁** | — | **✔ (risk–coverage)** |
| Voice | Speech Commands v2 | **Proposed CNN** | **Official speaker split, measured here** | **94.53 % accuracy** | 101,766 | **✔ (risk–coverage)** |

Three observations follow, and they are the reason we resist reading Table 13 as a leaderboard.

First, **the two blink rows are the clearest illustration of the paper's thesis**, and both are our own measurements rather than citations. The identical network scores 98.17 % under random row splitting and 92.36 % under subject-disjoint splitting (Table 9) — a 5.81 pp difference produced by nothing but the partition, on the same corpus and the same architecture. Any published figure obtained under random row splitting is therefore *not* comparable to the 93.67 % reported here in the ordinary sense; it is a different quantity, and it does not measure the property an assistive interface requires. That is why no such figure is tabulated beside ours.

Second, **the gaze rows show the asymmetric-exposure problem in situ**. The pre-activation baseline's ≈3.6° is its LOPO figure over fifteen participants, and its 3.621° here is measured on two held-out participants it *trained on*. By EDCP step 5 the latter is a lower bound on its true cross-person error, so the reported gap of 2.818° is conservative with respect to the proposed network. The literature offers no way to tighten this without either the baseline's training partition or a full leave-one-person-out retraining, which is precisely the access problem GERA is designed for.

Third, **the trades this paper claims rest only on rows measured here**. The proposed head-pose network's 7.367° at 1.14 M parameters and the baseline's 3.482° at 39.3 M parameters were produced in the same run, on identical crops, from one aligned array set, which is what makes the 34.4× / 2.12× trade of (296) a real measurement rather than citation arithmetic. Where the table says *not re-measured*, no claim of ours depends on that row.

---

## 6. Threats to validity

We state the limitations of this study explicitly and in descending order of how much they constrain the conclusions. Several were identified by the analysis itself and are recorded here rather than in the sections that produced them.

### 6.1 Held-out participant count for gaze

The participant-disjoint gaze split yields **two** held-out individuals. They contribute 64,172 samples, but by the effective-sample-size relation (10) the sample size for the cross-person question is the group count, not the row count:

$$
n_eff = n / [ 1 + (n̄ − 1) ρ ] → G/ρ = 2/ρ  as  n̄ → ∞ .
$$ (303)

Their fitted gains differ substantially — (0.586, 0.630) for p05 against (0.377, 0.430) for p07 by (219) — which is direct evidence that the between-participant variance component σ_b² in (7) is large relative to the within-participant component, and therefore that two participants is a small sample in the sense that matters.

This is a departure from established practice and should be read as such: leave-one-person-out over all fifteen participants is the standard protocol for person-independent evaluation on MPIIGaze, and it would give a substantially more robust estimate at fifteen times the compute cost. The single split was adopted under the compute constraint of Section 4.1, not because it is adequate. Two consequences follow. The gain dispersion of Fig. 18 should be treated as indicative rather than established, and the gaze accuracy figures throughout are point estimates from one fold whose fold-to-fold variability is unmeasured. The EDCP of Table 5 is independent of this choice — it prescribes *which groups may enter a comparison*, not *how many folds are run* — so the audit contribution is unaffected, but the gaze comparison itself would be strengthened by the full leave-one-person-out treatment.

### 6.2 The detector has one positive case

GERA is exercised on a negative control (head pose, Section 5.1.3) and on a single positive case: one corpus, fifteen groups, six outside the audited model's training partition. The Hanley–McNeil interval (50) on a 9-versus-6 group split is approximately

$$
AUC_r = 0.889 ,   95 % CI ≈ [ 0.70, 1.00 ] ,
$$ (304)

so the Mann–Whitney p = 0.006 establishes that the two group sets *differ*, not that the statistic separates membership at this rate elsewhere. The negative control constrains the false-positive direction; **nothing here constrains the false-negative direction**, since no case is available in which leakage is known to be present but mild. Its per-participant membership labels were reconstructed from the pipeline's group split and reproduce the strata of Table 10 to within 0.13°, so the AUC is indicative rather than exact.

The detector's derivation does not depend on this corpus — Proposition 2 requires only a reference model without differential familiarity — but its calibration is not established, and no threshold on r is proposed for that reason. Replication across corpora and modalities is the necessary next step, and until it exists r should be read as a screening flag that motivates the stratified decomposition of Theorem 3, never as a substitute for it.

### 6.3 Compute constraints and single-seed selection

Experiments ran on a consumer laptop with 8 GB of memory under intermittent CPU contention. Search trials used bounded subsets of 10,000–20,000 samples and the gaze configuration of Section 5.2.1 was trained on 80,000 of 427,316 available samples. Where budgets differ between compared configurations this is stated; the augmentation comparison of Section 5.2.1 uses an identical evaluation set for both configurations and is unaffected.

Each configuration was evaluated at **one seed**, so no variance over initialisations is available and no confidence interval on a single configuration's score can be quoted. Formally, the reported score is a single draw from the distribution over training randomness,

$$
R̂( f_{ϑ̂(σ)} ; ℰ ) ,   σ  fixed,  Var_σ[ · ]  unmeasured ,
$$ (305)

and differences below approximately 1 pp for classification or 0.2° for regression should not be treated as established without repetition. The *paired* intervals, effect sizes and correlations of Sections 5.2–5.3 are unaffected by this, since they characterise the difference between two fixed models over a sample population rather than the variability of training — a distinction worth making because those are the quantities on which the paper's conclusions rest.

### 6.4 Oracle bounds are not achievable

The affine recalibration of (197) and the per-sample model selection of (163) are fitted on the evaluation data and are therefore **upper bounds**. They are used only to close design questions negatively — recalibration hurts, fusion gains little — which is a use upper bounds support: if the best achievable version of an intervention is insufficient, every achievable version is. No positive claim rests on them.

### 6.5 Component accuracy is not system accuracy

Every figure in Section 5 measures a network in isolation, whereas the user experiences the composite Δ_m ∘ f_m of (1). Three specific gaps follow.

The deployed gaze path fits a per-user polynomial calibration downstream by (106)–(107), so 6.439° bounds the *uncalibrated* network rather than the cursor; the composed quantity is the residual scatter after the affine map, which this evaluation does not measure.

The dwell-averaging argument of (230)–(231) assumes consecutive-frame errors are less than perfectly correlated, and the run-integrity argument of (277)–(279) makes the same assumption. Neither is tested here. Both depend on the AR(1) persistence ϱ of (123), which is not identifiable from a per-frame corpus of independent crops and requires a live session to measure. Explicitly, the two conclusions hold for any ϱ < 1 in *direction* but their *magnitudes* scale as

$$
n_eff^run = n_run ( 1 − ϱ ) / ( 1 + ϱ ) ,
$$ (306)

so at ϱ = 0.8 the effective run length falls from 13 to 1.4 frames and the false-positive suppression of (277) weakens from 10⁻¹⁵ to approximately 10⁻². This does not reverse the recommendation of Section 5.5 — the false-negative asymmetry weakens in the same proportion — but it materially changes the safety margin, and it is the single measurement that would most improve confidence in the deployed thresholds.

End-to-end measurement — cursor error after calibration, click precision and recall in live sessions, gesture false-trigger rate — is the necessary next experiment, and no claim about deployed performance should be read into the component figures until it is done.

### 6.6 Latency measurement

Voice latencies come from one benchmark run over a 900-clip subset for the ASR baseline and the full test set for the proposed network. A separate full-test-set profiling run recorded 84.0 ms for the baseline; the conservative 26.5 ms figure is reported, so the stated 9.8× advantage is a **lower bound**. No per-sample baseline predictions were retained, so paired significance testing between the two voice systems was not possible and none is claimed.

### 6.7 Unscored providers

The geometric eye-aspect-ratio blink detector cannot be evaluated on the available corpora, which contain cropped eye regions without surrounding landmarks. No comparison is reported rather than an estimated one. This leaves the blink modality without any external reference point, which is why Section 5.5 is framed as a threshold-setting audit rather than as a comparison.

### 6.8 Deployment gap

All accuracies are measured on curated corpora. Live operation introduces face-detection failures, illumination extremes and eyewear that these corpora under-represent — and Fig. 53 identifies eyewear specifically as falling in the harder open-eye class by (281). Section 5.2.1 quantifies and partially removes one instance of this shift via the analytic augmentation family (34); the attribution evidence of Figs. 15, 30, 45 and 60 bounds the risk by establishing that the networks attend to anatomically and acoustically appropriate regions; neither eliminates it.

### 6.9 Generality of the framework

MAPLE is demonstrated on four modalities within one system. The formal results — Theorem 1, Proposition 2, Corollary 1, Theorem 3 — depend only on the existence of a grouping key, within-group dependence, and a reference model without differential familiarity, and are therefore not specific to vision or to this domain. But the *empirical* calibration of the detector, the choice of B = 1,000 bootstrap replicates, the reading of spread magnitude in preference to Q, and the sufficiency of a single reference model are all established on one comparison. Whether GERA behaves as well on text, tabular or time-series comparisons — where group structure is often weaker and the reference-model requirement harder to satisfy — is untested.

---

## 7. Conclusion and future work

### 7.1 Conclusion

The principal contribution of this work is a practical audit for **asymmetric training exposure between compared models**: a fault that resides in the comparison procedure rather than in any model's training pipeline, and that is therefore invisible to per-pipeline auditing. The hazard itself is documented in the foundation-model and benchmark-contamination literatures, but the standard leakage taxonomies are organised per pipeline and contain no category for it, while the existing remedies — lineage tracing and contamination search — presume documentation or corpus access that an external auditor rarely has.

Its three preconditions (Definition 2) are overlapping development corpora, group-structured data with within-group dependence, and an evaluation sample spanning at least one model's partition boundary. All three are common in appearance-based perception, because shared public corpora are the norm and identity is almost always the grouping key. Theorem 1 shows that under these conditions the reported score is a convex mixture of a memorisation and a generalisation regime whose weight is a sampling decision, and Corollary 1 shows the induced bias is signed and always favours the exposed model. In the system studied here the fault inflated one modality's apparent accuracy by 1.905°, of which 0.425°–1.068° (22 %–56 %) is attributable to training familiarity once the reference model prices the intrinsic difficulty of the affected stratum — the interval spanning the two difficulty models of (59), which the available data cannot distinguish, and under both of which difficulty fails to explain the whole gap. This is a decomposition (Theorem 3) that lineage analysis cannot supply at all, since knowing that a model saw a corpus does not say how much of its advantage came from having seen it.

**Two instruments make the fault tractable in practice.** The per-group error ratio r(g) detects it from predictions alone — the only assets an external audit reliably has are the two prediction streams and the group labels — separating training membership at AUC 0.889 (p = 0.006). It adapts difficulty calibration, developed for membership inference, from example level to group level, which is what makes it stable enough to read off published predictions; Proposition 2 states the condition under which the intrinsic-difficulty term cancels exactly. It is a screening statistic rather than a membership oracle, and unlike black-box contamination proofs it carries no false-positive guarantee. A negative control on head pose, where neither model saw the evaluation groups, returns I² = 0.0 % against 99.8 % for the leaked gaze comparison, establishing that it stays silent when it should even under a fourfold swing in intrinsic difficulty. The six-step exposure-disjoint comparison protocol of Table 5 then prevents the fault where the audit would otherwise have to catch it.

**Section 5.7 records what the correction was worth.** Under the uncorrected protocol the proposed gaze network scored level with its pretrained baseline and the decision was to keep it; under the corrected protocol the baseline is both more accurate and 7.1× smaller, and the decision reverses.

Evaluated under the corrected protocol, reproducible optimisation produced consistent gains, largest for voice (+9.81 pp) and head pose (−11.1 % MAE); domain-targeted augmentation modelling the analytic train–serve shift (34) reduced gaze angular error a further 9.5 %; and a controlled scaling experiment established that the residual is bounded by **training identity count rather than sample count** — the same group-structure property that makes the corpus leak-prone in the first place, by (222).

**The evaluation's principal analytical finding is that one mechanism explains the regression modalities' error structure.** Both networks operate in a shrinkage regime — fitted gains of 0.48 on both gaze axes and 0.46–0.49 on the two narrow-support head-pose axes — and that single parameter accounts for the isotropic radial growth of gaze error, the constant-ratio degradation of head-pose error, the axis-ranking inversions between absolute and variance-normalised metrics, and the absence of any localised failure mode. Inverting the contraction raises gaze angular error by 28.5 %, so the networks sit near their minimum-MSE shrinkage point and improvement requires a better *representation* rather than better *calibration*. The classification modalities are limited differently: ROC-AUC of 0.9836 and 0.9961 against operating-point accuracies of 93.67 % and 94.53 % place their residual error in decision logic, which is why a confidence gate and a threshold move — not additional capacity — are what improve them.

**Reading those results through the deployed decision rules changes three conclusions** that the component metrics alone would suggest. Head pose survives an 8.74° yaw MAE because its error is unbiased and its gesture rule averages ≈20 frames inside an 18° hysteresis band, while its 0.822 gain manifests as a stable, learnable requirement to turn ≈36.5° to trigger a 30° threshold. Blink inverts the conventional cost asymmetry: because a click requires ≈13 consecutive frames above threshold, isolated false positives cannot fire one, so false negatives — which break a deliberate closure run — are the more damaging frame-level error, and thresholds below the default 0.50 are the defensible setting. Voice contributes a finding that is a design fault rather than a model fault: the weakest keyword in the vocabulary, *go* at 0.877 precision, is bound to "resume control", the only command that reactivates a paused interface, and rebinding it to a phonetically distant token removes the largest error mode at zero computational cost.

**Measured against pretrained baselines on matched samples the proposed networks are less accurate, and this is reported without qualification:** 6.439° against 3.621° for gaze, 7.367° against 3.482° for head pose, 94.53 % against 95.50 % for voice. Their contribution is positional. For head pose and voice they hold competitive accuracy at 34× and 161× footprint reductions and an aggregate 6.68 ms per frame across three vision modalities, which is what permits four modalities to run concurrently in real time on commodity hardware without an accelerator. For gaze the pretrained alternative dominates on both accuracy and size, error correlation and oracle analysis show fusion cannot close the gap, and its adoption is recommended in place of the proposed network. Throughout, the attribution analysis establishes that the deployed networks compute the anatomically and acoustically appropriate operators, so the measured accuracies rest on identified transferable mechanisms rather than on dataset-specific shortcuts — which is the property a system intended for a user it has never seen must actually possess.

### 7.2 Future work

Six directions follow directly from the limitations of Section 6, ordered by how much each would strengthen the present conclusions.

**Replication of the detector across corpora and modalities.** GERA currently has one positive case and one negative control. The necessary next step is a set of comparisons spanning several corpora, several grouping keys, and — critically — cases in which leakage is known to be present but *mild*, which would constrain the false-negative direction that the present negative control cannot. A synthetic protocol in which the exposed fraction Π of (38) is varied continuously from 0 to 1, with everything else held fixed, would produce exactly the calibration curve r(Π) that the current study lacks and would permit a principled threshold on r to be proposed.

**Full leave-one-person-out gaze evaluation.** The single group-disjoint split of Section 6.1 should be replaced by leave-one-person-out over all fifteen MPIIGaze participants, which would convert the point estimates of Section 5.2 into fold-distributed estimates with measurable variance, and would supply fifteen rather than two observations of the identity-conditioned gain s(g) of (220). Since Section 5.2.4 identifies identity count as the binding constraint, this is also the experiment that would test that identification most directly.

**End-to-end measurement of the composite.** Section 6.5 is the largest gap between what is measured and what the user experiences. The three specific quantities required are: cursor error after per-user calibration and stabilisation, measured against known screen targets; click precision and recall in live sessions, which would also supply the AR(1) persistence ϱ of (123) that both the blink and head-pose arguments depend on; and gesture false-trigger rate under naturalistic head motion. All three are session-level measurements the deployed system already has the instrumentation to collect.

**Resolving the unexplained voice movement.** Section 5.1.1 records a 7.24 pp change in the voice figure between the published and fair-baseline columns despite an unchanged partition. Because the official Speech Commands split is already speaker-disjoint, this movement cannot be a leakage correction, and the present experiments do not identify its cause. Re-running the pre-correction voice configuration with the evaluation split logged explicitly would settle it in a single run, and until it is settled voice cannot serve as a second internal control alongside head pose — which is the reason the control in Section 5.1.1 rests on one modality rather than two.

**Multi-seed evaluation and variance reporting.** Replacing the single-seed protocol of Section 6.3 with a small ensemble of seeds per configuration would attach a variance to every reported score and would settle whether the sub-1-pp differences currently declared inconclusive are real.

**The vocabulary redesign the analysis prescribes.** Section 5.4 identifies a zero-cost intervention — rebinding "resume control" from *go* to a phonetically distant token with fricative or stop-cluster onset — predicted by (263) to remove at least 50 % of the total voice error mass. This prediction is testable at the cost of one evaluation run and would validate the entire decision-rule-aware framing, since it is a conclusion no component metric would have produced.

**Extending the exposure audit beyond pairwise comparison.** The present formulation compares two models. Realistic leaderboards compare many, with heterogeneous and undocumented exposure sets, and the natural generalisation is a matrix of pairwise ratios r_MM′(g) whose structure would identify *which* models in a set are exposed, not merely whether a given pair's comparison is contaminated. Whether the difficulty-cancellation of Proposition 2 survives that generalisation — in particular whether a consistent reference can be identified from the matrix itself rather than assumed — is an open theoretical question and, in our view, the most interesting one this work leaves behind.

---

## CRediT authorship contribution statement

**Mohammad Anas Shaik:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review and editing, Visualization.

## Declaration of competing interest

The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

## Data and code availability

All four corpora used in this study are publicly available and are cited in Table 4: the MRL Eye Dataset [@mrl], MPIIGaze [@mpiigaze][@mpiigaze_pami], 300W-LP and AFLW2000 [@w300lp][@aflw], and Google Speech Commands v2 [@speechcmd]. No new data were collected, and no human participants were recruited for this work.

The four trained networks are distributed as self-contained TorchScript modules [@torchscript], and the training, evaluation, hyperparameter-search and figure-generation scripts — together with the per-trial search logs and the aligned per-sample prediction arrays underlying every figure and table in Section 5 — are available from the author on reasonable request. The GERA audit of Algorithm 2 requires only two prediction streams and a group label per sample, so it can be reproduced against any pair of models without access to this system's weights.
