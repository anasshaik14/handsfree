---

## 4. Experimental settings

### 4.1 Implementation environment

All experiments ran on a consumer laptop with 8 GB of system memory and no discrete accelerator, under intermittent CPU contention from unrelated background processes. This is a deliberate constraint rather than an accident of availability: the target deployment is a machine the user already owns, and an evaluation conducted on hardware the deployment cannot assume would not answer the question the system poses. Two practical consequences of the constraint are recorded here and revisited in Section 6.

Training uses PyTorch [@pytorch] with intra-op parallelism pinned to two threads. The default thread count (eight on this machine) proved actively harmful: with eight-way intra-op parallelism across eight logical cores, any competing process turns the parallel region into kernel-dominated thread thrash, and a single trial that completes in 16 s on a quiet machine was observed to stall beyond 30 minutes under contention. Two threads is the setting at which parallelism still helps on a quiet machine but extra threads no longer add more context-switch overhead than compute. The *runtime* configuration differs and is pinned to a single thread, because per-frame inference at these model sizes is small enough that additional threads only spin; the same change reduced observed CPU utilisation of the deployed pipeline from approximately 400 % to below 100 % of one core.

Each trained model is exported as a self-contained TorchScript module [@torchscript] and loaded at runtime by `torch.jit.load`, so the deployed provider never imports the architecture definitions. Every latency and accuracy figure reported in Section 5 is measured on that exported artefact rather than on a research-time reimplementation, which is what permits Section 3.10's claim that the attribution explains what actually executes.

Search trials use bounded subsets of 10,000–20,000 samples to fit the memory budget; final configurations are retrained on the full training partition. The gaze configuration of Section 5.2.1 was trained on 80,000 of the 427,316 available normalised patches. Where budgets differ between compared configurations this is stated explicitly; the augmentation comparison of Section 5.2.1 uses an identical evaluation set for both configurations and is unaffected.

**Table 8.** Search budget, selected configuration and deployed size for each modality. Trial 0 of every search is pinned to the pre-correction deployed configuration.

| Modality | Trials | Optimiser | Learning rate | Weight decay | Dropout | Width / FC | Schedule | Aug. | Batch | Parameters | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Blink | 20 | AdamW | 1.09 × 10⁻³ | log-uniform | 0.467 | 16 / 64 | none | on | 128 | 136,066 | 0.57 |
| Gaze | 27 | AdamW | 2.93 × 10⁻⁴ | log-uniform | 0.182 | 32 / 128 | cosine | on | 256 | 551,810 | 2.23 |
| Head pose | 10 | AdamW | 1.32 × 10⁻³ | log-uniform | 0.204 | 32 / 128 | cosine | on | 64 | 1,142,339 | 4.60 |
| Voice | 10 | Adam | 2.93 × 10⁻³ | log-uniform | 0.499 | 32 / 64 | cosine | on | 128 | 101,766 | 0.44 |

### 4.2 Baselines

Each modality is compared against the strongest publicly available pretrained model that consumes the same input, with one exception recorded as an unscored comparison.

**Gaze.** A residual network [@resnet] with pre-activation blocks [@preact] trained on MPIIGaze [@mpiigaze_pami], which is the reference architecture for this benchmark. Its exposure set is all fifteen MPIIGaze participants, so by Definition 2 the comparison satisfies precondition 1 automatically and satisfies precondition 3 for any evaluation sample drawn from those fifteen. This is the comparison that motivated the entire analysis of Section 3.5, and EDCP step 5 applies to it: the baseline's held-out figure is reported as a lower bound on its true cross-person error.

**Head pose.** A continuous 6-D rotation-representation network [@sixdrep][@rot6d], the reference method for the 300W-LP → AFLW2000 protocol, at 39,323,398 parameters and 157.32 MB. Both models train on 300W-LP and score on AFLW2000, so *neither* has seen the evaluation identities. This is what makes head pose the negative control of Section 3.5.3, and the control is available at no additional experimental cost.

**Voice.** A production offline automatic-speech-recognition engine (Vosk small en-US [@vosk], Kaldi-derived [@kaldi], 70.90 MB), evaluated by matching its transcription against the six keywords. The engine possesses a native rejection option that the proposed classifier does not, which is why Section 5.4 reports the risk–coverage comparison of (133) rather than accuracy alone.

**Blink.** No pretrained neural comparison exists that consumes the same input. The practical alternative is the geometric eye-aspect-ratio rule [@ear], which requires the six palpebral landmarks per eye and therefore cannot be scored on 32 × 32 cropped eye regions containing no surrounding landmarks. **No comparison is reported rather than an estimated one.** Blink is therefore presented as a per-sample audit sufficient to set its operating threshold, which is the decision the evaluation must actually support.

### 4.3 Evaluation sets and matching

All comparisons are per-sample matched: both models score the identical evaluation samples, each receiving its own native preprocessing, and ground-truth alignment between prediction streams is verified numerically before any statistic is computed. Concretely, for each modality a single aligned array set (y_true, ŷ_A, ŷ_R) is generated once and every downstream figure and table is computed from it, which removes both metric-definition drift and sample-selection drift as competing explanations for a measured difference.

Evaluation set sizes are those of Table 4: 12,927 held-out eyes for blink (4,453 open, 8,474 closed); 64,172 held-out-participant patches for gaze; 1,969 AFLW2000 faces for head pose; 2,468 official-test clips for voice. For gaze, a 2,000-eye-per-participant cap inherited from the pretrained evaluation initially restricted the matched comparison to 4,000 samples; both networks were subsequently re-run over the full 64,172, and all figures reported here use the full set.

### 4.4 Statistical procedures

Paired sample-level statistics follow (159)–(161). Bootstrap variances for the heterogeneity analysis use B = 1,000 within-group resamples. Permutation p-values for the GERA separation statistic are exact by enumeration, which is feasible at G = 15 since the number of distinct membership assignments of the observed size is C(15, 6) = 5,005.

Three reporting conventions are adopted throughout and are stated here so that they are not mistaken for omissions.

First, **effect sizes and intervals accompany every significance claim**. With n in the tens of thousands, essentially any non-zero difference attains an arbitrarily small p-value, so a p-value alone carries no information about magnitude. Where a Wilcoxon p is quoted below 10⁻³⁰⁰ it should be read as "the sign of the difference is not in doubt", and the paired confidence interval and Cohen's d carry the substantive content.

Second, **medians accompany means for every right-skewed distribution**. All four modalities produce right-skewed error distributions, so the mean overstates the typical case; where the two diverge materially both are reported and the median is treated as the more representative summary.

Third, **oracle quantities are labelled as upper bounds and used only negatively**. The affine recalibration of Section 5.2.2 and the per-sample model selection of (163) are both fitted on the evaluation data. They are used exclusively to close design questions in the negative direction — recalibration hurts, fusion gains little — which is a use that upper bounds support; no positive claim rests on them.

### 4.5 Metric-formatting conventions

Angular quantities are reported in degrees to three decimal places where the paired interval justifies that precision and to two otherwise. Accuracies are reported as percentages to two decimal places, and differences between accuracies in percentage points (pp) to distinguish them from relative changes. Relative changes in regression error are reported as percentages of the baseline value. Where a metric is undefined for a provider — for example, parameter count for a closed-source ASR engine — the cell is left empty rather than filled with a zero, since a zero in such a cell would be read as a measurement.

### 4.6 Reproducibility

Every trial appends its complete configuration and validation score to a per-model trials file, so the search is reproducible from its logged rows. Initialisation, data shuffling and the group-partition shuffle are all seeded from a single value, so a configuration maps deterministically to a result. The group-partition routine carries a self-check asserting the three disjointness conditions of (35), coverage of the index set, and proximity of the realised row fractions to their targets, including under deliberately lopsided synthetic group sizes matching the worst case observed in MRL. The complete set of aligned per-sample arrays underlying every figure is written once and reused, so any figure can be regenerated without re-running inference.
