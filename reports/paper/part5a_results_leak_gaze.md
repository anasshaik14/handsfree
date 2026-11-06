---

## 5. Results and discussion

This section reports what evaluation under the MAPLE framework establishes. Section 5.1 quantifies both leakage faults and exercises the GERA detector with its negative control, since every subsequent number depends on those corrections being in place. Sections 5.2–5.5 report the four modalities in a fixed order — headline comparison, mechanism, diagnostic figures, synthesis figures, attribution — so that their conclusions can be compared rather than merely accumulated. Section 5.6 states what the four analyses jointly establish, and Section 5.7 reports the system-level accuracy–efficiency operating point and the architectural decision it produces.

### 5.1 Evaluation integrity

#### 5.1.1 Magnitude of the first-order leakage

Initial accuracies proved irreproducible under cross-subject conditions. The cause was structural rather than incidental: all four pipelines partitioned by random row sampling, and by (11)–(12) essentially every test row in these corpora then has a near-duplicate exemplar of the same individual available in training. Table 9 reports what replacing random row splitting by the group-disjoint partition of Algorithm 1 costs.

**Table 9.** Magnitude of the first-order (training-partition) leakage. "Published" is the original random-row protocol; "fair baseline" retrains the identical hyperparameters under group-disjoint partitioning.

| Modality | Published | Fair baseline | Change | Interpretation |
|---|---|---|---|---|
| Blink | 98.17 % accuracy | 92.36 % accuracy | −5.81 pp | 5.81 pp attributable to leakage |
| Gaze | 2.84° angular | 6.95° angular | +4.11° | Published figure scored the training partition |
| Head pose | 8.49° MAE | 8.29° MAE | −0.20° | Evaluation set already disjoint |
| Voice | 91.96 % accuracy | 84.72 % accuracy | −7.24 pp | Split already leak-free; partition unchanged, movement unexplained (see text) |

The correction is large precisely where the corpus admits leakage and small where it does not. Head pose is evaluated on AFLW2000, disjoint from 300W-LP by construction, and moves by 0.20°, well inside the seed-to-seed noise floor. It therefore serves as an **internal control** on the correction itself, in the precise sense of Section 3.4.2: group-disjoint partitioning suppresses a specific artefact rather than depressing scores indiscriminately, which is what licenses attributing the 5.81 pp and 4.11° corrections to leakage. A single-modality study could not have supplied this control.

The voice row requires a caveat that is stated here rather than glossed. Voice uses the official speaker-disjoint split, so its partition was already correct and **no partition change was applied to it** — yet its reported figure moves by 7.24 pp between the two columns. That movement therefore cannot be a leakage correction, and its cause is not established by the experiments reported here; the most likely explanation is a change in which split the published figure was computed on, but the original run was not instrumented to settle this and we do not assert it. Voice is consequently **not** usable as a second internal control, and only head pose is treated as one above. Resolving the discrepancy requires re-running the pre-correction voice configuration with the evaluation split logged explicitly, which Section 7.2 records as an open item.

Quantitatively, writing Δ_leak for the observed change and Δ_noise for the seed-to-seed floor, the attribution is justified when

$$
| Δ_leak | ≫ Δ_noise  for the affected modalities,  | Δ_leak | ≲ Δ_noise  for the controls,
$$ (168)

and Table 9 exhibits exactly this pattern with a separation of more than an order of magnitude on the regression modality.

Nothing in Section 5.1.1 is new. It is the standard remedy, correctly applied. The difficulty, and the subject of the remainder of Section 5.1, is that applying it correctly to every model in a comparison does not make the comparison correct.

#### 5.1.2 Asymmetric exposure: the stratified decomposition

After the correction of Section 5.1.1 every pipeline was group-disjoint and every model's individually reported score was sound. The gaze comparison was nevertheless still contaminated, for the reason Definition 2 states: the matched evaluation sampled equally from all fifteen MPIIGaze participants, nine of whom constitute the proposed network's training partition, while the pretrained baseline trained on all fifteen. All three preconditions hold — overlapping corpora, group structure with ρ > 0, and an evaluation sample spanning the audited model's partition boundary — so by Theorem 1 the aggregate is a quota-weighted mixture of two regimes.

!FIG protocol/P2_exposure_mechanism.png | Fig. 2. The mechanism of asymmetric exposure. Left: both models' training partitions drawn against the shared corpus's group population; the evaluation sample (dashed) spans the audited model's boundary but lies wholly inside the reference model's. Centre: by Theorem 1 the audited model's reported error is a convex mixture Ē_A(Π) = Π·Ē^in + (1−Π)·Ē^out whose weight is the exposed sampling mass Π, so the same models and the same data yield different reported errors under different quotas. Right: the per-group ratio r(g) = ē_A(g)/ē_R(g) cancels intrinsic difficulty (Proposition 2), leaving a quantity that separates the two strata.

Table 10 decomposes the consequence at matched sampling.

**Table 10.** Gaze angular error by partition membership, at matched sampling. The right-hand column is the group error ratio r of Definition 3.

| Evaluation subset | N | Proposed (A) | Pretrained (R) | ratio *r* |
|---|---|---|---|---|
| A's training participants (9) | 18,000 | 3.684° | 2.683° | 1.373 |
| A's validation participants (4) | 8,000 | 5.674° | 2.760° | 2.056 |
| **A's held-out participants (2)** | 4,000 | **6.494°** | **3.672°** | **1.769** |
| All fifteen participants | 30,000 | 4.589° | 2.835° | 1.619 |

The monotone progression 3.684° → 5.674° → 6.494° as training familiarity is withdrawn is the signature of the fault. Quantitatively, the audited model's error grows

$$
( 6.494 − 3.684 ) / 3.684 = 76.3 %
$$ (169)

across that boundary, while the reference model's moves

$$
( 3.672 − 2.683 ) / 2.683 = 36.9 %
$$ (170)

and stays within a degree throughout. The three-way progression also exhibits the structure that Proposition 2 predicts if the *validation* participants are treated as partially exposed — they were used for model selection, not for weight updates — with r rising from 1.373 on the fully exposed stratum to 2.056 on the selection-exposed stratum and settling at 1.769 on the wholly unexposed stratum. That the middle stratum carries the *highest* ratio is consistent with selection-induced overfitting to the validation groups compounding the direct familiarity effect, and it is a reminder that a validation partition is not an unexposed partition.

Applying Theorem 3 to Table 10, the total inflation of the aggregate relative to the held-out stratum is

$$
Ι = Ē_A^out − Ē_A^all = 6.494° − 4.589° = 1.905° ,
$$ (171)

the reference model's corresponding shift across the identical strata is

$$
Δ_R = Ē_R^out − Ē_R^all = 3.672° − 2.835° = 0.837° ,
$$ (172)

and the capability ratio on the unexposed stratum, which sets κ_max in (60), is

$$
κ_max = c_A/c_R = 6.494° / 3.672° = 1.769 .
$$ (173)

The familiarity component under the two difficulty models of (59) is therefore

$$
additive (κ = 1):   Φ = 1.905° − 0.837° = 1.068° ,   ϕ = 56.1 % ,
$$ (174)

$$
multiplicative (κ = κ_max):   Φ = 1.905° − 1.769 × 0.837° = 0.425° ,   ϕ = 22.3 % .
$$ (175)

**Both readings are reported, and the interval ϕ ∈ [22 %, 56 %] is the honest summary.** The two models are not distinguishable from the data available here: identifying κ would require observing at least one further model of known capability on the same strata, which this study does not have. It matters which is used — the additive reading attributes more than half the inflation to familiarity, the multiplicative reading less than a quarter — so quoting either alone would overstate the precision of the instrument. Where a single figure is needed below, the additive value is quoted because it is the reading the source analysis used, and it is flagged as the upper end of the interval each time.

The decomposition nevertheless does the work it was introduced for, and it does so under *both* readings. Without it, a sceptical reader may reasonably object that the held-out participants were simply harder, which would make the stratification an artefact rather than a fault. The reference model closes that objection quantitatively: harder they were — by 44 % of the observed gap under the additive model and by 78 % under the multiplicative one — but under neither does intrinsic difficulty account for all of it, and a familiarity component of at least 0.425° survives. That residual is the quantity lineage-based remedies cannot supply at all, since knowing *that* a model saw a corpus does not say how much of its advantage came from having seen it.

#### 5.1.3 The GERA detector and its negative control

An auditor examining a published comparison rarely knows how either model was partitioned. Fig. 3 reports the detector of Section 3.5.2 applied to this comparison, with the head-pose negative control alongside.

!FIG protocol/P1_leakage_detector.png | Fig. 3. Detecting asymmetric exposure without access to the split. (a) Per-participant mean angular error of the audited network against the reference model, which calibrates intrinsic group difficulty; the dashed line is parity. (b) The per-group ratio r(g), separated by whether the participant lies inside the audited network's training partition. (c) Negative control: r(g) normalised by its pooled value, for gaze grouped by participant (leakage present) against head pose grouped by yaw band (neither model saw the evaluation groups).

Panel (a) shows the structure the detector exploits. The two models agree closely on which participants are intrinsically easy — p00 to p03 sit near parity — while the audited network's error rises far above the reference's on participants it never trained on, displacing them *vertically* rather than along the diagonal. Vertical displacement is precisely the component intrinsic difficulty cannot explain, and it is the geometric content of Proposition 2: motion along the diagonal is d(g), motion off it is the familiarity term.

Panel (b) reduces that displacement to one number per group. Mean r is

$$
mean_{g ∈ 𝒯_A} r(g) = 1.369 ,   mean_{g ∉ 𝒯_A} r(g) = 2.072 ,
$$ (176)

giving by (47) an estimated familiarity discount

$$
φ̂_A = 1 − 1.369/2.072 = 0.339 ,
$$ (177)

so the audited network's error on groups it trained on is approximately 34 % lower than its error on groups it did not, after intrinsic difficulty has been priced out. The ratio separates membership at

$$
AUC_r = 0.889 ,   Mann–Whitney p = 0.006 ,
$$ (178)

with a Hanley–McNeil 95 % interval by (50) of approximately [0.70, 1.00] on a 9-versus-6 group split.

A detector is only as good as its behaviour when there is nothing to detect, and head pose supplies that control at no extra cost. Both the proposed network and the pretrained baseline train on 300W-LP and are scored on AFLW2000, so neither has seen the evaluation groups and r(g) must be homogeneous under any grouping. This is a demanding control rather than a trivial one, because intrinsic difficulty varies roughly fourfold across the pose manifold (Section 5.3): if the reference model failed to price that variation out, r would track it and the detector would fire on a corpus where no leakage exists.

Panel (c) reports the outcome. Grouping head pose by yaw band, r spans 1.96–2.28 — a spread ratio of

$$
spread_ctl = 2.28/1.96 = 1.16 ≈ 1.2×  about a pooled r̄ = 2.08 ,
$$ (179)

against 0.99–2.94 for gaze grouped by participant,

$$
spread_gaze = 2.94/0.99 = 2.97 ≈ 3.0× .
$$ (180)

Formally, applying (51)–(54),

$$
Head pose:  Q = 6.3  on  9 df,  p = 0.71,  I² = 0.0 % ,
$$ (181)

$$
Gaze:  Q = 5607  on  14 df,  p < 10⁻³⁰⁰,  I² = 99.8 % .
$$ (182)

The reference model therefore absorbs a fourfold swing in task difficulty while leaving the ratio flat, which is the assumption the detector rests on, tested on data where it could have failed.

One calibration caveat follows from the same experiment and is exactly the sample-size sensitivity that (55) predicts. Partitioning the gaze samples into fifteen *random* groups — destroying the participant structure — still yields

$$
I² = 41 %  at  p = 0.049 ,  despite an r range of only  1.56–1.69  (spread 1.08×) .
$$ (183)

At 2,000 samples per group a negligible effect reaches nominal significance because the inverse-variance weights grow linearly in n_g while the null dispersion does not. The magnitude of the spread, not its p-value, is therefore the informative readout: a 3.0× range against a 1.2× range is the contrast that matters, and no threshold on Q alone should be used to declare a comparison sound.

Two limitations must be stated with the result, consistent with Section 3.5.5. The separation is strong but imperfect: participant p10 sits at r = 2.18 despite being a training participant, so the statistic is a screening instrument rather than a membership oracle, and a single group's ratio should never be treated as proof. And the detector is inapplicable where no suitable reference exists — if every available model shares the audited model's partition, or if the reference itself has differential familiarity across groups, its across-group variation no longer isolates intrinsic difficulty. Unlike black-box contamination proofs [@oren], which yield a false-positive guarantee, no such guarantee is offered here; the detector's value is that it converts an unfalsifiable suspicion into a testable one at zero additional experimental cost, using only artefacts an external auditor already possesses.

#### 5.1.4 Optimisation under the corrected protocol

With the protocol repaired, optimisation could be attributed rather than merely observed. Table 8 gives the search budgets and selected configurations; Table 11 gives the effect.

**Table 11.** Effect of optimisation under the corrected protocol. "Fair baseline" is the pre-correction hyperparameter set retrained group-disjointly; "tuned" is the searched configuration under the same protocol.

| Modality | Metric | Fair baseline | Tuned | Change |
|---|---|---|---|---|
| Voice | Accuracy | 84.72 % | **94.53 %** | **+9.81 pp** |
| Head pose | MAE | 8.286° | **7.367°** | **−11.1 %** |
| Blink | Balanced accuracy | 90.88 % | **93.32 %** | **+2.44 pp** |
| Blink | Accuracy | 92.36 % | **93.67 %** | **+1.31 pp** |
| Gaze | Angular error, held-out | 7.114° | **6.439°** | **−9.5 %** |

The composition of the selected configurations tests the prediction of (104). Augmentation was selected in four searches of four and decoupled weight decay in three of four, while the chosen widths stayed modest:

$$
ℙ[ aug selected ] = 4/4 ,   ℙ[ AdamW selected ] = 3/4 ,   median( selected width ) = 32 = midpoint of 𝒲 .
$$ (184)

This degeneracy is informative rather than incidental. Once the split stops rewarding memorisation, the binding term in the excess-risk decomposition (104) migrates from approximation error to estimation error, so configurations that suppress identity-specific co-adaptation dominate configurations that add capacity. The pre-correction hyperparameters were not poorly chosen for the task they were *scored* on — they were well chosen for a task that included identity retrieval. For a system whose users are by definition unseen, that is the distinction that matters.

Every blink metric improved except false-negative rate, which rose 1.20 pp while false-positive rate fell 6.09 pp. Whether that exchange is favourable depends entirely on the decision rule of Table 6, and Section 5.5 shows the answer is not the obvious one. The tuned configurations of Table 11 are the models carried into Sections 5.2–5.5; all figures from this point describe them.

### 5.2 Gaze — cursor positioning

Gaze drives the OS cursor, but never directly: the deployed path fits a 16-point per-user polynomial calibration (105)–(107) over the network's output, corrects it online by recursive least squares (108)–(110) on every deliberate click, then smooths with a one-euro filter (112)–(115) and a 12 px deadzone (116). The quantity that governs cursor quality is therefore the residual scatter that survives an affine map, not the raw angular error — a distinction that turns out to be the central finding of this section rather than a caveat on it.

#### 5.2.1 Domain-targeted augmentation

Gaze transferred worst of the four modalities, and the cause is a deterministic covariate shift rather than sampling noise. Training consumes pre-normalised, illumination-corrected eye patches produced by the warp (16), while inference receives landmark-cropped regions from a consumer webcam. Writing 𝒟_train and 𝒟_serve for the two input distributions, the discrepancy is not a sampling difference but a systematic transformation

$$
𝒟_serve = T_shift # 𝒟_train ,   T_shift = Blur_{σ_b} ∘ Jitter_{α,β} ∘ Rot_δ ∘ Crop_ℓ ,
$$ (185)

where # denotes pushforward. The augmentation policy [@augsurvey] was therefore extended to model that shift *analytically* rather than generically, per (34): Gaussian blur for sensor resolution, brightness and contrast jitter for illumination, bounded ±8° in-plane rotation for residual head roll, alongside horizontal flipping with yaw-sign inversion. Scored on all 64,172 held-out samples with both configurations on the identical sample set,

$$
e_ang:  7.114° → 6.439°  ( −9.49 % ) ,   MAE:  4.569° → 4.135°  ( −9.50 % ) .
$$ (186)

Because every member of (34) is label-preserving up to the flip correction, the gain cannot be additional supervision; it reflects an enlarged effective input manifold at zero parameter and zero data cost, which for an 8 GB training budget is the only kind of improvement available. That both a robust statistic (MAE) and a non-robust one (mean angular error) improved by the *same* 9.5 % is diagnostic. Writing the error distribution before and after as F₀ and F₁, a policy that trimmed a tail would satisfy

$$
F₁ ≼_st F₀  only in the upper quantiles,  so  Δmean ≫ Δmedian  and  ΔMAE ≠ Δe_ang ,
$$ (187)

whereas a uniform rescaling F₁(x) = F₀(x/c) satisfies

$$
Δ𝔼 / 𝔼 = Δ median / median = ΔMAE / MAE = 1 − c
$$ (188)

for every location statistic simultaneously. The measured equality of the two relative changes to within 0.01 pp is therefore the signature of removing a distributional mismatch — rescaling the whole error distribution — rather than repairing isolated failures.

#### 5.2.2 Comparison against the pretrained baseline, and its mechanism

Against the MPIIGaze pre-activation residual baseline on all 64,172 held-out samples, and applying EDCP steps 3–4 so that only the two exposure-disjoint participants enter, the proposed network attains 6.439° mean angular error (median 5.873°, p90 11.180°) against 3.621° (median 2.985°, p90 6.876°). The paired statistics of (159)–(161) give

$$
d̄ = 2.818° ± 0.026°  (95 %) ,  s_d = 3.300° ,  Cohen's d = 0.854 ,  Wilcoxon p < 10⁻³⁰⁰ ,
$$ (189)

with the proposed network nearer the truth on only

$$
ℙ[ e_A < e_R ] = 18.86 %
$$ (190)

of samples. The gap is large, unambiguous and not a tail artefact. Per EDCP step 5, the baseline's figure is a lower bound on its true cross-person error, so the reported contrast is conservative with respect to the proposed network by (61).

**The mechanism.** What generates the gap is measurable. Fitting the shrinkage regression (154) axis-wise gives

$$
ŝ_pitch^A = 0.4832 ± 0.0021 ,   ŝ_yaw^A = 0.4812 ± 0.0016 ,
$$ (191)

$$
ŝ_pitch^R = 0.7491 ,   ŝ_yaw^R = 0.8257 ,
$$ (192)

so the proposed network reproduces less than half the true angular excursion on both axes. By (155) this is predictive shrinkage — the response of an MSE-trained estimator to irreducible uncertainty in its own representation, which contracts predictions toward the conditional mean because contraction is variance-optimal, the same admissibility argument that underlies Stein estimation [@stein][@jamesstein]. Inverting (158) with the empirical target variance yields the implied representational noise

$$
σ̂_ε² / σ_θ² = ( 1 − ŝ ) / ŝ = ( 1 − 0.482 ) / 0.482 = 1.075 ,
$$ (193)

that is, the network's internal estimate carries noise power slightly exceeding the signal power of the target itself. The seven diagnostic figures that follow each test a consequence of that single parameter, and Section 5.3 shows the same mechanism governing head pose.

Fig. 4 resolves the aggregate error into its angular structure.

!FIG gaze/G1_error_sky.png | Fig. 4. Error sky — polar heat map of mean angular error against gaze direction and eccentricity, proposed network versus pretrained baseline.

The proposed network's error field brightens with radius but is uniform in azimuth, exactly as shrinkage requires. Under gain s the systematic component of the prediction error is

$$
𝔼[ θ̂ − θ | θ ] = ( s − 1 ) θ  ⟹  ‖ 𝔼[ θ̂ − θ | θ ] ‖ = ( 1 − s ) ‖θ‖ ,
$$ (194)

a function of eccentricity ‖θ‖ alone with no dependence on the azimuth arg(θ). The symmetry simultaneously excludes the competing hypothesis: a geometric fault in cropping or normalisation would introduce an azimuth-dependent term and necessarily break isotropy. The consequence for the deployed system is concrete — no per-direction calibration lookup can compensate an isotropic radial law, so the fix must be an overall scale correction, which is precisely what the polynomial calibration (106) applies.

Fig. 5 separates the systematic component from the stochastic one.

!FIG gaze/G2_displacement_field.png | Fig. 5. Displacement field — vectors from true gaze to mean prediction across the gaze plane; vector length is systematic bias rather than scatter.

Additive bias is +1.987° in pitch and +0.829° in yaw against +0.473° and +0.334° for the baseline. Decomposing the per-axis mean squared error into bias and variance,

$$
MSE_a = bias_a² + var_a ,   bias share_a = bias_a² / MSE_a ,
$$ (195)

gives

$$
bias share_pitch = 20.5 % ,   bias share_yaw = 1.9 % .
$$ (196)

Removing both offsets exactly reduces angular error from 6.439° to 6.023°, a 6.5 % gain — real but minor. The field therefore adjudicates between two candidate defects and rejects the simpler one: additive miscalibration is present and measurable but is not the principal fault. The multiplicative gain deficit is, and only a transform with a scale term can address it.

That claim is directly testable by inverting the fitted gain, and (157) predicts the outcome. Rescaling each axis by its fitted slope and intercept,

$$
θ̂_corr = ( θ̂ − ĉ ) / ŝ ,
$$ (197)

an oracle correction fitted on the evaluation data itself and hence an upper bound on any achievable recalibration, *increases* angular error

$$
6.439° → 8.272° ,  a  +28.5 %  degradation ,
$$ (198)

exactly as (157) requires: restoring the correct scale also restores the full amplitude √(σ_ε²) of the representational noise that shrinkage was suppressing. Substituting the measured ŝ = 0.482 into (156), the predicted relative penalty of moving from s* to unity is

$$
MSE(1)/MSE(s*) = 1/s* = 2.07  in MSE terms,  i.e.  √2.07 = 1.44  in RMS terms,
$$ (199)

and the observed 1.285× increase in the *chordal* angular statistic (20) is consistent with this once the difference between the RMS and arccos metrics is accounted for. The contraction is thus not a defect but the estimator's rational response to its own uncertainty; the network sits near its minimum-MSE shrinkage point.

This resolves an apparent contradiction in the deployed design and is the most practically consequential result of the section. The system fits a per-user polynomial and corrects it online, which necessarily restores scale and therefore necessarily amplifies noise — and downstream of that calibration sit a one-euro filter and a 12 px deadzone whose function is to suppress exactly the amplified high-frequency component. The pipeline is thus already structured around this trade: calibration buys angular scale, temporal filtering pays for it in bandwidth. Formally, the deployed chain implements

$$
cursor = Deadzone_R ∘ OneEuro_{f_min, β} ∘ Calib_{𝐰} ∘ f_gaze ,
$$ (200)

in which Calib multiplies both signal and noise by approximately 1/ŝ ≈ 2.07 and OneEuro attenuates the noise by a factor depending on its spectral content, with the low deployed minimum cut-off f_min = 0.3 Hz chosen for precisely this reason. The 6.439° figure is consequently a *pessimistic* estimate of deployed cursor error, and the honest statement is that it bounds the uncalibrated network rather than the shipped system, whose end-to-end accuracy this evaluation does not measure.

Fig. 6 expresses the comparison in the tolerance terms an interface designer budgets against.

!FIG gaze/G3_tolerance_sunburst.png | Fig. 6. Tolerance sunburst — nested rings giving the proportion of predictions within 5°, 10° and 15° of ground truth for each model.

Writing the tolerance-hit function as

$$
H(t) = ℙ[ e_ang ≤ t ] ,
$$ (201)

the measured values are

$$
H_A(5°) = 39.47 % ,  H_R(5°) = 78.51 % ,  ratio  1.99 ,
$$ (202)

$$
H_A(10°) = 85.23 % ,  H_R(10°) = 96.81 % ;   H_A(15°) = 97.14 % ,  H_R(15°) = 99.44 % .
$$ (203)

Convergence under relaxed tolerance is again the shrinkage signature: by (194) contraction damages predictions in proportion to their distance from the mean, so its cost concentrates in the tight-tolerance regime and the ratio H_R(t)/H_A(t) decreases monotonically in t. For the interface this sets the boundary between viable and non-viable use before calibration — dense target selection is out of reach, coarse region selection at a 10–15° budget is not — and it makes that boundary *predictable from the gain* rather than something to be discovered by user testing.

Fig. 7 asks whether the deficit is uniform or sample-dependent, which decides whether running both networks could ever be worthwhile.

!FIG gaze/G4_win_skyline.png | Fig. 7. Head-to-head skyline — every held-out sample sorted by its signed per-sample error gap between the two models.

The proposed network wins 18.86 % of 64,172 paired comparisons and its wins are smaller in magnitude than its losses. More decisive is the error correlation of (162),

$$
r_AR = 0.491 ,   ρ_{AR} = 0.360 ,
$$ (204)

a substantial shared factor attributable to intrinsic sample difficulty, meaning the models are not complementary. The oracle bound (163)–(164) gives

$$
Ē_oracle = 3.303° ,   Gain_max = 1 − 3.303/3.621 = 8.8 % ,
$$ (205)

far too little to justify the cost of a second network in a pipeline already budgeted at 6.68 ms per frame by (165). This closes the fusion question quantitatively, and it is the evidence behind the recommendation of Section 5.7.

Fig. 8 applies the Bland–Altman construction [@blandaltman], the appropriate instrument when two estimators of one physical quantity are compared.

!FIG gaze/G5_bland_altman.png | Fig. 8. Bland–Altman density — per-axis bias and 95 % limits of agreement against ground truth, over a sample-density field.

Limits of agreement, defined as

$$
LoA_a = bias_a ± 1.96 · σ_a ,
$$ (206)

bound 95 % of individual predictions and are what the cursor controller must absorb:

$$
LoA_pitch^A ≈ [ −5.68°, +9.66° ] ,  LoA_yaw^A ≈ [ −10.91°, +12.57° ] ,
$$ (207)

$$
LoA_pitch^R ≈ [ −4.62°, +5.56° ] ,  LoA_yaw^R ≈ [ −6.87°, +7.54° ] .
$$ (208)

The pitch interval's asymmetry is the additive bias of Fig. 5 made visible, and the yaw interval is 1.63× wider than the baseline's — on the axis carrying most interaction-relevant motion in a seated user, which is why the deployed one-euro filter is tuned with a low minimum cut-off that trades response speed for steadiness at fixation. Neither density field fans, indicating homoscedasticity:

$$
Var[ θ̂ − θ | θ ] ≈ const  in θ ,
$$ (209)

so a variance-stabilising reparameterisation would not help; the error is a constant multiplicative effect, not a signal-dependent one.

Fig. 9 stratifies by eccentricity, the variable shrinkage identifies as governing.

!FIG gaze/G6_eccentricity_ridgeline.png | Fig. 9. Error versus eccentricity — angular-error distributions stratified by how far off-centre the true gaze lies.

Both models degrade off-axis, which has an optical basis — foreshortening and eyelid occlusion reduce informative pixels per unit of angular change — but the ridgelines separate by a broadly constant factor rather than diverging. That distinction discriminates between two accounts. A purely optical limit acts additively on both models,

$$
e_M(‖θ‖) = e_M(0) + κ_opt(‖θ‖)  ⟹  e_A − e_R  constant in ‖θ‖ ,
$$ (210)

whereas a gain deficit acts multiplicatively by (194),

$$
e_M(‖θ‖) ≈ ( 1 − s_M ) ‖θ‖ + ν_M  ⟹  e_A / e_R  approximately constant in ‖θ‖ .
$$ (211)

The data support (211), so the proposed network has no peripheral failure mode to repair — only a uniformly contracted response, which is good news for the screen-edge behaviour the deployed 1.2 expansion factor is designed to reach.

Fig. 10 exposes the axis asymmetry that pooled error conceals.

!FIG gaze/G7_true_vs_pred.png | Fig. 10. Predicted versus true — density-weighted regression fits with coefficient of determination for pitch and yaw.

By (153) the proposed network attains R² = 0.302 in pitch and 0.561 in yaw against 0.748 and 0.836. Because its fitted gain is essentially identical on both axes by (191), the R² disparity cannot be a gain effect. Writing (153) in terms of the shrinkage model,

$$
R²_a = ŝ_a² σ_{θ_a}² / ( ŝ_a² σ_{θ_a}² + σ_{ζ_a}² ) ,
$$ (212)

equal ŝ_a on both axes forces the disparity into the residual variance σ_ζ². The physical explanation is direct: vertical gaze is signalled through eyelid aperture and iris occlusion, cues confounded with blink state and individual eyelid morphology, whereas horizontal gaze produces a clean identity-invariant lateral translation of the iris. Pitch is thus the axis most contaminated by identity-specific appearance and the one expected to suffer most under a nine-identity training set — a prediction the attribution analysis of Fig. 15 below confirms mechanistically. It also explains why the deployed vertical gain (3.4) is set higher than would be needed for a symmetric estimator.

Figs. 11–14 re-render the same per-sample data as continuous fields. They add no measurement, but three of the claims above — isotropy, the dominance of incoherent over coherent displacement, and residual anisotropy — are claims about *joint* structure and are tested most directly here.

!FIG figures_art/gaze/GA1_comet_field.png | Fig. 11. Comet field — every held-out sample drawn as a streak from its true gaze direction to the model's prediction; streak length is the per-sample angular error.

Retaining every displacement rather than their mean separates coherent from incoherent error. Decomposing the displacement field,

$$
𝐮(θ) = θ̂ − θ = 𝔼[ θ̂ − θ | θ ]  +  ( θ̂ − 𝔼[ θ̂ | θ ] ) ,
                 ⏟ coherent            ⏟ incoherent
$$ (213)

the streaks share a common upward coherent component of 1.987° but their residual directions are largely uncorrelated. Coherent displacement is calibration error and is removable by the deployed polynomial; incoherent displacement is representational noise and is not, passing instead to the one-euro filter. The visual dominance of the incoherent component is the geometric reason exact bias removal recovers only 6.5 % by (196), and it predicts that most of the deployed system's gaze quality is bought by temporal smoothing rather than by calibration.

!FIG figures_art/gaze/GA2_aurora.png | Fig. 12. Aurora of error — the angular-error distributions of both models rendered as luminous density curtains.

The curtains overlap substantially despite a 2.818° mean separation — the graphical form of d = 0.854, large but far from disjoint, and consistent with the 18.86 % win rate. Both distributions are right-skewed (means 6.439° and 3.621° against medians 5.873° and 2.985°), with skew ratios

$$
mean/median:  6.439/5.873 = 1.096  (A) ,   3.621/2.985 = 1.213  (R) ,
$$ (214)

so neither model's typical behaviour is captured by its mean, and median-based reporting is the safer convention for this task. The skew is proportionally greater for the baseline, indicating its advantage is realised most strongly on the bulk of easy samples — precisely where a correctly-scaled estimator gains over a contracted one, since by (194) the contracted estimator's systematic error vanishes only at θ = 0.

!FIG figures_art/gaze/GA3_nebula.png | Fig. 13. Gaze nebula — every held-out gaze as a point placed by direction and eccentricity, with brightness encoding its error.

Collapsing the stratifications of Figs. 4 and 8 into one field makes the governing law testable rather than inferred: brightness rises with radius and is uniform in angle. Error is a function of eccentricity alone with no azimuthal dependence — the exact signature of the (1 − s)‖θ‖ term of (194) and the single strongest confirmation that the gaze deficit is a resolution limit rather than a geometric fault, since any crop or normalisation error would introduce azimuthal asymmetry.

!FIG figures_art/gaze/GA4_error_islands.png | Fig. 14. Error islands — density contours of predictions relative to ground truth for each model; a tight island centred on the cross is accurate and unbiased.

The proposed network's island is displaced by the 1.987° pitch offset and elongated in yaw at a standard-deviation ratio

$$
σ_yaw/σ_pitch = 5.988/3.914 = 1.53  (A) ,   3.675/2.596 = 1.42  (R) .
$$ (215)

The anisotropy is therefore partly intrinsic to the task and partly amplified by the proposed network — consistent with the horizontally-tuned operator identified next, since an operator optimised for horizontal displacement necessarily places its widest residual along the horizontal axis, and consistent in turn with the deployed controller using different horizontal and vertical gains.

#### 5.2.3 What the network computes

Accuracy establishes that a network is right; it does not establish that it is right for a transferable reason. Attribution follows Section 3.10: Integrated Gradients [@ig] against a zero baseline over 32 steps, computed directly on the deployed TorchScript module, with the input partitioned into a 4 × 4 grid by (139) so that each region assumes the role of a feature. Completeness (138) makes the decompositions exact rather than indicative.

!FIG attribution/gaze/A1_summary.png | Fig. 15. Gaze — global attribution summary for predicted yaw over the 36 × 60 eye patch, regions ranked by mean absolute attribution.

The two dominant regions, R2C3 and R2C2, sit side by side across the vertical midline carrying opposite-signed attribution. The network has therefore learned a **signed differential operator** across the midline,

$$
F_yaw(x) ≈ Σ_{k ∈ left} w_k ⟨x, ϕ_k⟩ − Σ_{k ∈ right} w_k ⟨x, ϕ_k⟩ ,   w_k > 0 ,
$$ (216)

which is exactly the computation yaw estimation requires, since horizontal gaze is encoded by iris displacement relative to the eye-corner frame. Attribution concentrates in the horizontal band with little vertical structure, independently explaining the axis asymmetry of Fig. 10: capacity was allocated horizontally, so pitch is served by whatever vertical signal survives incidentally. That the operator is *differential* rather than absolute is what makes it robust to the illumination variation a home webcam delivers, since a differential operator annihilates the additive component of the photometric family that (15) normalises.

!FIG attribution/gaze/A2_waterfall.png | Fig. 16. Gaze — waterfall attribution for two extreme instances of predicted yaw.

Extreme instances are carried to their predictions by opposing contributions from the same midline pair, confirming the differential structure operates per instance rather than emerging from averaging. The residual regions contribute a small consistently-signed offset — the attribution-space counterpart of the +1.987° bias of Fig. 5 — which localises that bias in a distributed background term rather than in the discriminative operator, and explains why removing it recovers only 6.5 % of angular error.

!FIG attribution/gaze/A3_similarity.png | Fig. 17. Gaze — global explanation under similarity ordering.

Similarity ordering produces a smooth continuum rather than discrete blocks, which is correct for regression over a continuous target: neighbouring gaze angles should induce neighbouring explanations. Formally, the explanation map is approximately Lipschitz,

$$
‖ Attr(·; x₁) − Attr(·; x₂) ‖ ≤ L ‖ θ(x₁) − θ(x₂) ‖ ,
$$ (217)

with no discontinuity anywhere along the ordering. This rules out piecewise or identity-conditioned strategy switching, so the elevated error of Section 5.2.2 is uniform representational imprecision rather than a mixture of a good rule and a bad one — the attribution-level statement of the isotropic error field of Fig. 4, reached by an independent route.

!FIG attribution/gaze/A4_ordering.png | Fig. 18. Gaze — global explanation under dataset ordering across held-out participants p05 and p07.

Explanation structure stays coherent across the participant boundary while accuracy does not:

$$
e_ang(p05) = 5.786° ,   e_ang(p07) = 7.138° ,
$$ (218)

$$
( ŝ_pitch, ŝ_yaw )(p05) = ( 0.586, 0.630 ) ,   ( ŝ_pitch, ŝ_yaw )(p07) = ( 0.377, 0.430 ) .
$$ (219)

The same operator is applied to both at substantially different gain, which localises identity sensitivity in the operator's *calibration* rather than its *form*. Writing the per-participant model as

$$
θ̂ = s(g) · 𝒪(x) + c(g) ,   𝒪 fixed across g ,  s(g) varying ,
$$ (220)

this is the mechanism behind the ceiling established next, and it is also the strongest available justification for the deployed per-user calibration stage: if identity enters as a scalar gain, a per-user scalar fit is the structurally correct remedy, and (106)–(107) is exactly such a fit.

#### 5.2.4 The ceiling is set by identity count, not sample count

To test whether the residual error was data-limited, the network was retrained on 250,000 samples (175,199 training frames, 3.1× the configuration of Section 5.2.1). Validation error did not improve:

$$
6.32°  at  3.1× data  against  5.81°  at  1× ,
$$ (221)

a null result — indeed a slight regression within run-to-run variation. The explanation follows directly from the effective-sample-size relation (10). Additional frames from the same nine identities are strongly correlated with those already present, so they raise n but not G, and by (10)

$$
n_eff = n / [ 1 + (n̄ − 1) ρ ] → G / ρ  as  n̄ → ∞ ,
$$ (222)

which is bounded above by a constant depending only on the identity count. Extra frames therefore tighten the empirical risk estimate without enlarging the set of appearance modes the hypothesis must cover. Fig. 18 supplies the mechanism — what nine identities fail to span is the *range of gain* s(g) in (220), not the *form* of the operator 𝒪 — and this explains why augmentation succeeded where more data failed: augmentation synthesises new imaging conditions by (34), while further frames of the same nine people supply none.

For the project this converts an open-ended tuning effort into a closed question. No further optimisation on MPIIGaze will close the gap to the pretrained baseline; only a corpus with more participants would, and Section 5.7 therefore recommends adopting the pretrained model rather than continuing to tune. It also predicts that gaze alone among the four modalities would benefit disproportionately from more participants rather than more frames — a prediction the other three sections test implicitly, since none of them exhibits the same identity–gain coupling.
