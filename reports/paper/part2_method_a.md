---

## 3. Methodology

This section presents MAPLE (Multimodal Audited Perception with Leakage-aware Evaluation), a framework for building and honestly evaluating a multi-model perception stack whose components are compared against third-party pretrained baselines. As illustrated in Fig. 1, MAPLE comprises five integrated phases: (1) group-structured corpus preparation, in which every corpus is assigned an explicit grouping key and partitioned group-disjointly; (2) modality-specific preprocessing and compact network training under a seeded, pinned-baseline random search; (3) *exposure auditing*, in which the comparison — not merely each pipeline — is tested for asymmetric exposure using the prediction-only GERA statistic and its negative control; (4) decision-rule-aware evaluation, in which each network's error is scored against the statistic its deployed decision rule makes diagnostic; and (5) mechanistic attribution on the deployed modules, establishing that measured accuracy rests on transferable operators. Phase 3 is the novel component and is developed in full in Section 3.5; the remainder formalises the system it audits.

!FIG protocol/P0_framework.png | Fig. 1. Overview of the MAPLE framework. Four group-structured corpora are partitioned by identity, feeding four compact task-specific networks trained under a pinned-baseline random search. Before any cross-model comparison is reported, the comparison itself is audited for asymmetric exposure by the GERA statistic against a reference model, with a negative control on a corpus where exposure is impossible by construction; only exposure-disjoint strata enter the reported contrast. Each network's error is then read through the deployed decision rule that consumes it, and Integrated-Gradients attribution on the shipped TorchScript modules identifies the operator each network computes.

### 3.1 Problem formalisation

#### 3.1.1 The interaction system as a composition

Let the system observe, at discrete time *t*, a camera frame **I**ₜ ∈ ℝ^(H×W×3) and an audio buffer **a**ₜ ∈ ℝ^L. A *modality* m ∈ ℳ = {blink, gaze, pose, voice} is a pair (f_m, Δ_m) consisting of a learned perception map f_m and a deterministic decision rule Δ_m. Writing 𝒞_m for the modality-specific crop or transform applied to the raw observation, the per-frame computation is

$$
u_m,ₜ = f_m(𝒞_m(𝐈ₜ, 𝐚ₜ) ; ϑ_m),    aₘ,ₜ = Δ_m(u_m,ₜ, u_m,ₜ₋₁, …, u_m,ₜ₋K ; ψ_m)
$$ (1)

where ϑ_m are the learned parameters, ψ_m the decision-rule parameters (thresholds, dwell times, hysteresis bands, filter cut-offs), and a_m,ₜ ∈ 𝒜_m the emitted action, with 𝒜_m possibly containing a null action. The essential structural point, and the one that governs the entire evaluation design of Section 3.11, is that the user experiences the composite Δ_m ∘ f_m and never f_m alone. Consequently the loss under which f_m is *trained* and the statistic by which f_m is *usefully reported* need not coincide, and in three of the four modalities studied here they do not.

The complete system output at time *t* is the arbitrated union

$$
𝒮(𝐈ₜ, 𝐚ₜ) = ⋃_{m ∈ ℳ} Δ_m(f_m(𝒞_m(𝐈ₜ, 𝐚ₜ))),
$$ (2)

subject to an arbitration relation that suppresses conflicting simultaneous actions (Section 3.9.5).

#### 3.1.2 The concurrency constraint

The three vision modalities share a single face-landmark inference [@mediapipe][@facemesh] ℓ(**I**ₜ), executed once per frame, whose output supplies the crops 𝒞_blink, 𝒞_gaze and 𝒞_pose. Writing τ(·) for wall-clock latency, real-time operation at frame rate F requires

$$
τ(ℓ) + Σ_{m ∈ ℳ_vis} [ τ(𝒞_m) + τ(f_m) + τ(Δ_m) ] + τ(stab) ≤ 1 / F
$$ (3)

with ℳ_vis = {blink, gaze, pose} and τ(stab) the cursor stabilisation chain. At F = 30 Hz the budget is 33.3 ms. The voice modality runs on an independent audio thread and enters (3) only through CPU contention, not through the frame-serial path. Inequality (3) is the constraint that makes the *sum* of model costs the binding quantity, and therefore the constraint under which an individual model's accuracy may rationally be traded.

#### 3.1.3 Risk, and why the empirical risk is the wrong target here

For a modality with sample space 𝒳_m × 𝒴_m, distribution 𝒟_m and loss ℓ_m, the population risk of a hypothesis f is

$$
R_m(f) = 𝔼_{(x,y) ∼ 𝒟_m} [ ℓ_m(f(x), y) ]
$$ (4)

estimated on a finite sample S = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ by

$$
R̂_m(f ; S) = (1/n) Σ_{i=1}^{n} ℓ_m(f(xᵢ), yᵢ).
$$ (5)

The estimator (5) is unbiased for (4) only when the sample is drawn i.i.d. from 𝒟_m. In every corpus used here it is not: samples are organised into groups (a subject, a participant, a base face identity, a speaker) with strong within-group dependence, and the deployment distribution places the *entire* group of a new user outside the training data. The relevant risk is therefore the group-conditional one,

$$
R_m^grp(f) = 𝔼_{g ∼ 𝒢} 𝔼_{(x,y) ∼ 𝒟_m(· | g)} [ ℓ_m(f(x), y) ],
$$ (6)

where 𝒢 is the population of groups. Equations (5) and (6) coincide only if the evaluation groups are disjoint from the training groups; Sections 3.4 and 3.5 are concerned with the two distinct ways in which that condition can fail.

#### 3.1.4 Notation

Table 3 fixes the notation used throughout. Modality subscripts are dropped wherever the context is unambiguous.

**Table 3.** Summary of mathematical notation.

| Symbol | Description |
|---|---|
| ℳ, ℳ_vis | Set of modalities; the three vision modalities sharing one landmark inference |
| f_m(· ; ϑ_m) | Learned perception map of modality *m* with parameters ϑ_m |
| Δ_m(· ; ψ_m) | Deployed decision rule consuming f_m, with parameters ψ_m |
| 𝒞_m | Modality-specific crop / transform of the raw observation |
| 𝒢, g | Population of groups; a single group (subject, participant, face identity, speaker) |
| γ(i) ∈ 𝒢 | Group label of sample *i* |
| 𝒯_A, 𝒯_R | Training-partition group sets of the audited and reference models |
| 𝒱, ℰ | Validation and evaluation index sets |
| A, R | Audited model and reference (control) model |
| e_A(i), e_R(i) | Per-sample error of the audited and reference model on sample *i* |
| ē_A(g), ē_R(g) | Group-mean error of A and R on group *g* |
| r(g) | Group error ratio ē_A(g) / ē_R(g) — the GERA statistic |
| d(g) | Intrinsic difficulty of group *g* |
| c_A, c_R | Model-level capability constants |
| φ | Familiarity discount enjoyed by a model on its own training groups |
| ρ | Intraclass correlation of the loss within a group |
| n, n_g, G | Sample count; samples in group *g*; number of groups |
| n_eff | Effective sample size after accounting for within-group dependence |
| π | Per-group sampling quota vector of the evaluation sample |
| s | Fitted regression gain of predicted on true target (shrinkage gain) |
| θ, θ̂ | True and predicted target (angle, in radians or degrees) |
| σ², σ_ε² | Target variance; irreducible representational noise variance |
| Q, I² | Cochran's heterogeneity statistic and the derived inconsistency index |
| α_OE, β, f_c | One-euro filter smoothing factor, speed coefficient, cut-off |
| **w**, **P**, λ_f | Calibration weights, RLS inverse-covariance, forgetting factor |
| T_hold, T_dwell, T_cd | Blink hold threshold, gesture dwell time, cooldown |
| θ_on, θ_off | Gesture trigger and release thresholds (hysteresis band) |
| 𝒜_m | Action alphabet of modality *m* |

### 3.2 Corpora, grouping keys and evaluation sets

Four public corpora supply the four modalities. Each is assigned an explicit grouping key, and that assignment is the precondition for everything that follows: a corpus with group structure and within-group dependence is exactly a corpus in which both leakage faults of Section 3.4 and Section 3.5 can occur.

**Table 4.** Corpora, grouping keys and evaluation sets.

| Modality | Training corpus | Evaluation set | N | Grouping key | Groups |
|---|---|---|---|---|---|
| Blink | MRL Eye Dataset (84,898 greyscale/IR eye crops) | Held-out subjects | 12,927 | Subject identity | 37 |
| Gaze | MPIIGaze, normalised patches (427,316) | Held-out participants | 64,172 | Participant | 15 |
| Head pose | 300W-LP (~122,450 synthesised renders) | AFLW2000, untouched | 1,969 | Base face identity | 3,837 |
| Voice | Speech Commands v2 (18,657 clips, 6 keywords) | Official test list | 2,468 | Speaker | official |

#### 3.2.1 Within-group dependence and effective sample size

Let ℓᵢ = ℓ(f(xᵢ), yᵢ) denote the per-sample loss and let γ(i) be the group label. Model the loss by a one-way random-effects decomposition

$$
ℓᵢ = μ + b_{γ(i)} + εᵢ,    b_g ∼ (0, σ_b²),  εᵢ ∼ (0, σ_w²),
$$ (7)

with b and ε independent. The intraclass correlation of the loss [@kish] is then

$$
ρ = σ_b² / (σ_b² + σ_w²),
$$ (8)

and for a sample containing G groups of equal size n̄ = n/G the variance of the sample mean loss is inflated relative to the i.i.d. case by the design effect

$$
Var(ℓ̄) = (σ_b² + σ_w²)/n · [ 1 + (n̄ − 1) ρ ],   DEFF = 1 + (n̄ − 1) ρ,
$$ (9)

so that the effective sample size is

$$
n_eff = n / [ 1 + (n̄ − 1) ρ ].
$$ (10)

Equation (10) is the quantitative statement of why a corpus with thousands of frames per participant does not supply thousands of independent observations, and why the number of *groups* rather than the number of *rows* is the sample size that governs a cross-person claim. For the gaze evaluation of Section 5.2, n = 64,172 rows drawn from G = 2 held-out participants gives n_eff → 2/ρ̄ as n̄ grows, which is why Section 6 records the held-out participant count as the principal limitation of the study notwithstanding the large row count.

#### 3.2.2 The leakage probability under random row splitting

Consider partitioning n rows uniformly at random into a training fraction p and a test fraction 1 − p, ignoring group structure. For a test row i belonging to a group of size n_g, the probability that at least one row of the same group falls in the training partition is

$$
ℙ[ leak(i) ] = 1 − (1 − p)^{n_g − 1} ,
$$ (11)

which for the corpora of Table 4 is numerically indistinguishable from unity: with p = 0.7 and n_g = 2,300 (MRL), 1 − 0.3^2299 < 10^−1200. Under random row splitting essentially *every* test row has a near-duplicate exemplar of the same individual available in training, and the expected fraction of the test set that is contaminated in this sense is

$$
𝔼[ contaminated fraction ] = (1/n) Σ_{i=1}^{n} [ 1 − (1 − p)^{n_{γ(i)} − 1} ] ≈ 1.
$$ (12)

The measured consequence is reported in Section 5.1: replacing random row splitting by group-disjoint splitting costs 5.81 pp of blink accuracy and 4.11° of gaze angular error, while leaving head pose and voice — whose evaluation sets are group-disjoint by construction — unchanged to within the seed-to-seed noise floor.

### 3.3 Preprocessing and input representations

Each modality applies a deterministic transform 𝒞_m from raw observation to network input. All transforms are implemented identically in the training scripts and in the deployed providers, with the single documented exception noted in Section 5.2.1.

#### 3.3.1 Eye-region extraction and photometric normalisation (blink)

Given the landmark set ℒ(**I**) returned by the shared face-landmark inference, the eye region is the axis-aligned bounding box of the palpebral landmark subset ℒ_eye, expanded by a margin κ and resampled to the canonical size 32 × 32:

$$
𝐛 = [ min_x ℒ_eye − κ w, min_y ℒ_eye − κ h, max_x ℒ_eye + κ w, max_y ℒ_eye + κ h ],
$$ (13)

$$
x_blink = Resize_{32×32} ( Gray ( 𝐈 [ 𝐛 ] ) ) / 255 ∈ [0,1]^{32×32}.
$$ (14)

Photometric normalisation applies per-crop standardisation, which removes the additive and multiplicative illumination terms that dominate variation across sessions and devices:

$$
x̃ = ( x − μ_x ) / ( σ_x + ϵ ),   μ_x = (1/|x|) Σ_{u} x_u,  σ_x² = (1/|x|) Σ_{u} (x_u − μ_x)²,
$$ (15)

with ϵ = 10⁻⁶ guarding degenerate crops. Section 5.5 shows via attribution that the network subsequently learns a *contrast-polarity* code over the central fissure region, which is precisely the representation for which (15) is the correct normalisation, since a polarity code is invariant to the affine photometric family that (15) quotients out.

#### 3.3.2 Gaze patch normalisation and angular parameterisation

MPIIGaze supplies eye patches already normalised by the standard procedure, which applies a virtual camera rotation **R**_n and scaling **S**_n such that the eye centre lies on the optical axis at fixed distance and the eye's horizontal axis is image-aligned:

$$
𝐌 = 𝐒_n 𝐑_n,   x_gaze = Warp_{𝐌} ( Gray ( 𝐈 ) ) ∈ ℝ^{36×60}.
$$ (16)

The corpus annotates a 3-D gaze unit vector **g** = (g_x, g_y, g_z)ᵀ in the normalised camera frame, converted to the pitch–yaw parameterisation the network regresses:

$$
𝐠̂ = 𝐠 / ( ‖𝐠‖₂ + ϵ ),
$$ (17)

$$
θ_pitch = arcsin( − ĝ_y ),   θ_yaw = arctan2( − ĝ_x , − ĝ_z ).
$$ (18)

The inverse map, used to recover a direction from a prediction for angular-error computation, is

$$
𝐠(θ_pitch, θ_yaw) = ( − cos θ_pitch sin θ_yaw ,  − sin θ_pitch ,  − cos θ_pitch cos θ_yaw )ᵀ.
$$ (19)

Angular error between a prediction and ground truth is computed on the sphere rather than in the parameter plane, since the parameter plane is not isometric:

$$
e_ang = arccos ( clip_{[−1,1]} ( 𝐠(θ̂) · 𝐠(θ) ) ) · 180/π  [degrees].
$$ (20)

Equation (20) is the statistic reported throughout as "angular error"; the separately reported "MAE" is the mean absolute error in the (pitch, yaw) parameter plane,

$$
MAE = (1/2n) Σ_{i=1}^{n} ( |θ̂_pitch,i − θ_pitch,i| + |θ̂_yaw,i − θ_yaw,i| ) · 180/π,
$$ (21)

and the two are reported together because they respond differently to the shrinkage mechanism of Section 5.2.2: (20) is a chordal quantity that grows with eccentricity, (21) a per-axis quantity that does not.

#### 3.3.3 Face crop and head-pose parameterisation

The head-pose network consumes a square RGB crop derived from the 68-point landmark set, expanded to include the full facial envelope:

$$
c = ( (min_x + max_x)/2 , (min_y + max_y)/2 ),   ℓ_side = (1 + κ_f) · max( max_x − min_x , max_y − min_y ),
$$ (22)

$$
x_pose = Resize_{64×64} ( 𝐈 [ c ± ℓ_side/2 ] ) ∈ ℝ^{3×64×64}.
$$ (23)

Ground-truth pose is supplied as intrinsic Euler angles (pitch, yaw, roll) = (θ_p, θ_y, θ_r), which define the rotation

$$
𝐑 = 𝐑_z(θ_r) 𝐑_y(θ_y) 𝐑_x(θ_p),
$$ (24)

with the usual elementary rotations

$$
𝐑_x(a) = [[1,0,0],[0,cos a,−sin a],[0,sin a,cos a]],  𝐑_y(a) = [[cos a,0,sin a],[0,1,0],[−sin a,0,cos a]],  𝐑_z(a) = [[cos a,−sin a,0],[sin a,cos a,0],[0,0,1]].
$$ (25)

The network regresses (θ_y, θ_p, θ_r) in degrees directly. Samples with max(|θ_y|, |θ_p|, |θ_r|) > 99° are excluded, following the standard protocol, because beyond that range the projected facial evidence is largely self-occluded and the annotation itself becomes unreliable; Section 5.3 shows that error nevertheless grows sharply well inside this bound and identifies the occlusion mechanism responsible.

Horizontal flip augmentation is *not* label-preserving for this task, and applying it naively is a silent correctness fault. The correct paired transform is

$$
( x , (θ_y, θ_p, θ_r) ) ↦ ( flip_h(x) , (−θ_y, θ_p, −θ_r) ),
$$ (26)

and Section 5.3 reports the diagnostic that confirms it was applied correctly: the per-band error distributions broaden and shift together rather than becoming bimodal, whereas sign confusion from an uncorrected flip would produce bimodality.

#### 3.3.4 Log-mel acoustic front end

Audio is resampled to F_s = 16 kHz and reduced to a fixed one-second window by right-padding or truncation:

$$
a′[n] = a[n] · 𝟙[ n < min(L, F_s) ] ,  n = 0, …, F_s − 1.
$$ (27)

The short-time Fourier transform [@welch] uses N_fft = 400 (25 ms) with hop H = 200 (12.5 ms), giving T = 81 frames:

$$
X[k, τ] = Σ_{n=0}^{N_fft − 1} a′[ τH + n ] · w[n] · exp( −j 2π k n / N_fft ),
$$ (28)

with w a Hann window. The power spectrum is projected onto B = 40 mel-spaced triangular filters,

$$
mel(f) = 2595 · log₁₀( 1 + f/700 ),
$$ (29)

$$
H_b[k] = triangular filter b, centred at
mel⁻¹( mel(f_low) + b · ( mel(f_high) − mel(f_low) ) / (B+1) ),
$$ (30)

$$
M[b, τ] = Σ_{k} H_b[k] · | X[k, τ] |² ,
$$ (31)

and compressed logarithmically to give the network input

$$
x_voice[b, τ] = log( M[b, τ] + ϵ ) ,  ϵ = 10⁻⁶ ,  x_voice ∈ ℝ^{1 × 40 × 81}.
$$ (32)

The logarithm in (32) is what makes the representation approximately invariant to recording gain, since a multiplicative gain becomes an additive offset that the first convolution's bias and the subsequent batch normalisation absorb. Section 5.4 shows via attribution that the network learns an implicit voice-activity gate over the time axis of (32) — concentrating attribution on central time slots and discarding the silence padding — which is the property that permits transfer to a live microphone stream whose padding statistics differ from the corpus.

#### 3.3.5 The augmentation operator family

Augmentation is modelled as a distribution over label-preserving (or label-covariantly-transformed) operators. Writing 𝒜 = {T_ω : ω ∼ p_ω} for the family, training minimises the augmented risk

$$
R̂^aug(f ; S) = (1/n) Σ_{i=1}^{n} 𝔼_{ω ∼ p_ω} [ ℓ( f(T_ω(xᵢ)), 𝒯_ω(yᵢ) ) ] ,
$$ (33)

where 𝒯_ω is the induced label transform — the identity for blink and voice, the sign flip of (26) for head pose, and yaw negation for gaze. The operator family used for the gaze domain-shift correction of Section 5.2.1 is constructed to model the *specific* deterministic shift between normalised training patches and landmark-cropped inference regions, and is therefore analytic rather than generic:

$$
T_ω = Rot(δ) ∘ Jitter(α, β) ∘ Blur(σ_b) ,   δ ∼ 𝒰(−8°, +8°),  α ∼ 𝒰(0.8, 1.2),  β ∼ 𝒰(−0.2, 0.2),  σ_b ∼ 𝒰(0, σ_max),
$$ (34)

with Jitter(α, β) applying x ↦ clip(α x + β), Blur(σ_b) a Gaussian kernel modelling sensor resolution loss, and Rot(δ) an in-plane rotation modelling residual head roll. Because every member of (34) is label-preserving up to the flip correction, any gain it produces cannot be additional supervision; it is an enlargement of the effective input manifold at zero parameter and zero data cost. Section 5.2.1 measures a 9.5 % reduction in *both* a robust and a non-robust error statistic, which is the signature of removing a distributional mismatch rather than repairing isolated failures.

### 3.4 First-order leakage: group-disjoint partitioning

#### 3.4.1 The partition constraint

Let the index set be [n] = {1, …, n} with group labels γ : [n] → 𝒢. A three-way partition ([n] = 𝒯 ⊎ 𝒱 ⊎ ℰ) is *group-disjoint* if

$$
γ(𝒯) ∩ γ(𝒱) = ∅,  γ(𝒱) ∩ γ(ℰ) = ∅,  γ(𝒯) ∩ γ(ℰ) = ∅,
$$ (35)

that is, no group contributes rows to more than one part. Under (35) the evaluation estimator (5) is an unbiased estimate of the group-conditional risk (6) restricted to the evaluation groups, whereas under random row splitting it estimates a mixture that Section 3.5.2 makes explicit.

Because group sizes are highly unequal — MRL subjects span roughly 700 to 10,000 frames — a partition that is group-disjoint need not be close to the target row fractions. We therefore deal groups greedily in decreasing size order into whichever part is furthest below its row target, which is the classical longest-processing-time-first heuristic for multiway number partitioning [@graham]:

$$
part(g) = arg max_{j ∈ {1,2,3}} ( n · φ_j − c_j ),   c_j ← c_j + n_g ,
$$ (36)

with φ = (0.70, 0.15, 0.15) the target fractions and c_j the running row count of part *j*. The heuristic admits the standard bound: for group sizes n_(1) ≥ n_(2) ≥ … the maximum deviation of any part from its target satisfies

$$
| c_j − n φ_j | ≤ max_g n_g ,
$$ (37)

which is tight in the worst case but far from tight in practice; the realised deviations for all four corpora are below 5 % of n. Algorithm 1 states the procedure.

```algorithm
Algorithm 1 — Group-disjoint partitioning (greedy longest-first deal)
Require: group labels γ(1..n); target fractions φ = (φ₁, φ₂, φ₃) with Σφⱼ = 1; seed σ
Ensure: index sets 𝒯, 𝒱, ℰ satisfying the disjointness constraint (35)
 1: 𝒰 ← unique(γ);  rng ← RandomState(σ);  shuffle(𝒰, rng)
 2: for all g ∈ 𝒰 do n_g ← |{ i : γ(i) = g }| end for
 3: targets ← (n·φ₁, n·φ₂, n·φ₃);  counts ← (0, 0, 0);  buckets ← (∅, ∅, ∅)
 4: for all g ∈ sort(𝒰, key = −n_g) do                    ▷ largest group first
 5:     j ← arg max_{j∈{1,2,3}} ( targets[j] − counts[j] )  ▷ furthest below target
 6:     buckets[j] ← buckets[j] ∪ {g};  counts[j] ← counts[j] + n_g
 7: end for
 8: 𝒯 ← { i : γ(i) ∈ buckets[1] };  𝒱 ← { i : γ(i) ∈ buckets[2] };  ℰ ← { i : γ(i) ∈ buckets[3] }
 9: assert γ(𝒯) ∩ γ(𝒱) = γ(𝒱) ∩ γ(ℰ) = γ(𝒯) ∩ γ(ℰ) = ∅
10: assert 𝒯 ⊎ 𝒱 ⊎ ℰ = [n]
11: return 𝒯, 𝒱, ℰ
```

The grouping keys are, per corpus: the subject identifier encoded in the MRL filename; the participant identifier p00–p14 for MPIIGaze; the base face identity for 300W-LP, recovered by stripping the synthesised view index and flip tag from the filename so that all renders of one face share a key; and, for Speech Commands, the official validation and testing lists, which are already speaker-aware and therefore require no custom grouping.

#### 3.4.2 What the correction is worth, and the internal control

Three quantities are reported per modality throughout this work: the *published* score obtained under the faulty random-row protocol; the *fair baseline*, retraining the original hyperparameters under the corrected protocol; and the *tuned* score under that same corrected protocol. Separating the three is what prevents the protocol effect and the optimisation effect from being conflated, a conflation that would otherwise make it impossible to attribute any reported improvement.

Two of the four modalities have evaluation sets that are group-disjoint by construction and receive no partition change at all — head pose is evaluated on AFLW2000, disjoint from 300W-LP by design, and voice uses the official speaker-disjoint split — so either could in principle serve as an *internal negative control* on the correction itself. If group-disjoint partitioning merely depressed scores indiscriminately, an unchanged partition would nonetheless fall along with the changed ones. Section 5.1.1 reports that head pose moves by less than the seed-to-seed noise floor and therefore does serve as that control, which is what licenses attributing the blink and gaze corrections to leakage rather than to a stricter protocol in general; a single-modality study could not supply it. It also reports that the voice figure moves by 7.24 pp despite an unchanged partition, for reasons this study does not establish, and voice is therefore **not** relied on as a second control. The control rests on head pose alone, and the reader should weigh it accordingly.

### 3.5 Second-order leakage: asymmetric exposure between compared models

After the correction of Section 3.4 every pipeline in this work is group-disjoint and every model's individually reported score is sound. The *comparison* between them is nevertheless still contaminated, because the fault does not live in either pipeline. It lives in the join.

#### 3.5.1 Formalisation

**Definition 1 (Exposure set).** For a model M evaluated on a sample drawn from group population 𝒢, the *exposure set* 𝒯_M ⊆ 𝒢 is the set of groups whose data entered M's training partition. A group g is *exposed* to M if g ∈ 𝒯_M.

**Definition 2 (Asymmetric exposure).** A comparison between models A and R on an evaluation sample ℰ with group labels γ(ℰ) ⊆ 𝒢 is *asymmetrically exposed* when the following three conditions hold simultaneously:

1. **(Overlapping development corpora)** A and R were developed on corpora with non-empty intersection at the group level: 𝒯_A ∩ 𝒯_R ≠ ∅ or both derive from a common corpus;
2. **(Group structure with within-group dependence)** the corpus carries a grouping key with intraclass loss correlation ρ > 0 in the sense of (8);
3. **(Boundary-spanning evaluation sample)** the evaluation sample is drawn from groups on both sides of at least one model's partition boundary: γ(ℰ) ∩ 𝒯_A ≠ ∅ **and** γ(ℰ) \ 𝒯_A ≠ ∅ (or the same with 𝒯_R).

Condition 3 is the operative one and the one that per-pipeline auditing cannot see: it is a statement about the relation between the evaluation sample and *the other model's* partition, and neither pipeline's own audit has any representation of that partition.

**Theorem 1 (Mixture representation and quota sensitivity).** Let A be a model with exposure set 𝒯_A, and let the evaluation sample allocate a fraction π_g of its mass to group g, with Σ_g π_g = 1. Define the exposed mass

$$
Π = Σ_{g ∈ γ(ℰ) ∩ 𝒯_A} π_g  ∈ (0, 1).
$$ (38)

Then the reported aggregate error of A decomposes as the convex combination

$$
Ē_A(π) = Π · Ē_A^in + (1 − Π) · Ē_A^out,
$$ (39)

where Ē_A^in and Ē_A^out are the exposed- and unexposed-stratum means, each formed by averaging ē_A(g) over the groups of its own stratum under the weights π_g renormalised to sum to one. Consequently

$$
∂ Ē_A / ∂ Π = Ē_A^in − Ē_A^out ,
$$ (40)

and whenever Ē_A^in ≠ Ē_A^out the reported error is a strictly monotone function of the sampling quota Π alone. In particular, a change to the per-group sampling quota changes the reported accuracy without changing any model, any datum, or any metric definition.

*Proof.* Write the sample mean error as a group-weighted mean, ē_A = Σ_g π_g ē_A(g). Split the index set of groups by membership in 𝒯_A and factor the exposed mass Π out of the first sum and (1 − Π) out of the second; this yields (39) directly, since the renormalised weights within each stratum sum to one by construction. Differentiating (39) in Π, holding the two conditional means fixed, gives (40). Monotonicity is immediate from the sign of (40). ∎

Theorem 1 states precisely what is objectionable. Under Definition 2 the aggregate is a mixture of a *memorisation* regime and a *generalisation* regime, with the mixing weight fixed by a sampling decision rather than by the research question. Two honest evaluators applying the same metric to the same models on the same corpus can report different numbers, and neither is making an error in the ordinary sense.

**Corollary 1 (Bias of the reported contrast).** Let the quantity of interest be the true cross-group contrast Δ* = Ē_A^out − Ē_R^out, and let the reported contrast be Δ̂ = Ē_A(π) − Ē_R(π). If R has no differential familiarity across γ(ℰ) — that is, Ē_R^in = Ē_R^out — then

$$
Δ̂ − Δ* = Π · ( Ē_A^in − Ē_A^out ) ≤ 0,
$$ (41)

so the reported contrast understates A's true disadvantage by exactly the exposed mass times the within-model familiarity gap. The bias is signed, and it always favours the exposed model.

*Proof.* Substitute (39) for both models into Δ̂, apply Ē_R^in = Ē_R^out so that Ē_R(π) = Ē_R^out, and subtract Δ*. The sign follows from Ē_A^in ≤ Ē_A^out, which holds whenever training exposure does not increase error. ∎

Corollary 1 is the reason Section 5 reports every comparison on the exposure-disjoint stratum only, and the reason a baseline trained on the whole corpus must be reported as a *lower bound* on its own cross-person error rather than as a point estimate (protocol step 5, Section 3.5.6).

#### 3.5.2 The GERA statistic

An auditor examining a published comparison rarely knows how either model was partitioned. The detector proposed here requires only what such an audit does have: the two prediction streams over a shared evaluation set, and a group label per sample.

**Definition 3 (Group error ratio).** For each group g ∈ γ(ℰ), let

$$
ē_A(g) = (1/n_g) Σ_{i : γ(i) = g} e_A(i),   ē_R(g) = (1/n_g) Σ_{i : γ(i) = g} e_R(i),
$$ (42)

and define the **group error ratio**

$$
r(g) = ē_A(g) / ē_R(g).
$$ (43)

The role of R is to calibrate intrinsic group difficulty. Some groups are simply harder than others — a fact A's own errors cannot separate from familiarity — and a reference model with no *differential* familiarity across those groups supplies exactly that missing control, because its across-group variation is intrinsic difficulty alone.

**Proposition 2 (Exact cancellation of intrinsic difficulty).** Suppose per-group errors follow the multiplicative difficulty model

$$
ē_M(g) = d(g) · c_M · ( 1 − φ_M · 𝟙[ g ∈ 𝒯_M ] ) ,   M ∈ {A, R},
$$ (44)

with d(g) > 0 the intrinsic difficulty of group g, c_M > 0 a model-level capability constant, and φ_M ∈ [0,1) the familiarity discount M enjoys on its own training groups. Then

$$
r(g) = ( c_A / c_R ) · ( 1 − φ_A 𝟙[ g ∈ 𝒯_A ] ) / ( 1 − φ_R 𝟙[ g ∈ 𝒯_R ] ),
$$ (45)

which is free of d(g). If in addition R has no differential familiarity across γ(ℰ) — either 𝒯_R ⊇ γ(ℰ) or 𝒯_R ∩ γ(ℰ) = ∅, so the indicator is constant — then

$$
r(g) = κ · ( 1 − φ_A 𝟙[ g ∈ 𝒯_A ] ),   κ = ( c_A / c_R ) / ( 1 − φ_R 𝟙[γ(ℰ) ⊆ 𝒯_R] ) ,
$$ (46)

so r(g) takes exactly two values, κ(1 − φ_A) on exposed groups and κ on unexposed groups, and the ratio of the two group means identifies the familiarity discount:

$$
φ̂_A = 1 − ( mean_{g ∈ 𝒯_A} r(g) ) / ( mean_{g ∉ 𝒯_A} r(g) ).
$$ (47)

*Proof.* Substituting (44) into (43), the factor d(g) appears in numerator and denominator and cancels identically for every g, giving (45). Under the stated condition on 𝒯_R the denominator indicator is constant over γ(ℰ) and may be absorbed into κ, giving (46). Equation (47) follows by taking means over the two strata and forming the ratio. ∎

Proposition 2 is the formal content of the phrase "the reference prices the intrinsic difficulty". It also states the detector's precondition sharply: R must have *constant* exposure across the evaluation groups. If R is differentially familiar — exposed to some evaluation groups and not others — its across-group variation no longer isolates intrinsic difficulty and (45) does not reduce.

In practice (44) holds only approximately, so r(g) is dispersed rather than two-valued, and the detector is a *screening* statistic. We assess separation by the rank statistic

$$
AUC_r = ℙ[ r(G_out) > r(G_in) ] = (1/(n_in n_out)) Σ_{g ∈ 𝒯_A} Σ_{h ∉ 𝒯_A} 𝟙[ r(h) > r(g) ] + ½ 𝟙[ r(h) = r(g) ],
$$ (48)

with the Mann–Whitney U statistic [@mannwhitney] U = n_in n_out AUC_r and the exact two-sided permutation p-value

$$
p = ℙ_{H₀}[ | U − 𝔼U | ≥ | u_obs − 𝔼U | ],   𝔼U = n_in n_out / 2,
$$ (49)

evaluated by enumeration when the group counts are small, as they are here. Confidence on AUC_r uses the Hanley–McNeil variance estimate [@hanley]

$$
Var(AUC_r) = [ AUC(1−AUC) + (n_in − 1)(Q₁ − AUC²) + (n_out − 1)(Q₂ − AUC²) ]
/ ( n_in n_out ),
$$ (50)

with Q₁ = AUC/(2 − AUC) and Q₂ = 2AUC²/(1 + AUC).

#### 3.5.3 Heterogeneity testing and the negative control

A detector is only as good as its behaviour when there is nothing to detect. We therefore test whether r(g) is *homogeneous* across groups, using the standard random-effects machinery. Let v(g) denote the bootstrap variance [@efron] of r(g), obtained by resampling within group g,

$$
v(g) = Var_{b=1..B} [ r^(b)(g) ],   r^(b)(g) = ē_A^(b)(g) / ē_R^(b)(g),
$$ (51)

and let w(g) = 1/v(g) be the inverse-variance weight. The pooled estimate and Cochran's heterogeneity statistic [@cochran] are

$$
r̄ = Σ_g w(g) r(g) / Σ_g w(g),
$$ (52)

$$
Q = Σ_g w(g) ( r(g) − r̄ )² ,   Q ∼ χ²_{G−1} under H₀ : r(g) ≡ r̄ ,
$$ (53)

with the derived inconsistency index [@higgins]

$$
I² = max( 0 , ( Q − (G − 1) ) / Q ) · 100 %.
$$ (54)

I² is the proportion of observed dispersion in r not explained by within-group sampling error. Under the null of no differential familiarity, and with a reference satisfying Proposition 2's condition, r(g) should be homogeneous and I² ≈ 0.

The negative control exploits a structural property of the head-pose corpora and costs nothing: both the proposed network and the pretrained baseline train on 300W-LP and are scored on AFLW2000, so neither has seen the evaluation groups, and r(g) must be homogeneous under *any* grouping of the evaluation set. It is a demanding control rather than a trivial one, because intrinsic difficulty varies roughly fourfold across the pose manifold (Section 5.3); if the reference failed to price that variation out, r would track it and the detector would fire on a corpus where no leakage exists. Section 5.1.3 reports the outcome.

One calibration caveat follows from the same machinery and bears on how (53)–(54) should be read. Cochran's Q is sensitive to sample size: at large n_g a negligible effect reaches nominal significance, because w(g) ∝ 1/v(g) grows linearly in n_g while the null dispersion does not. Formally, for a fixed true dispersion δ² = Var_g[r(g)] in excess of sampling error,

$$
𝔼[Q] ≈ (G − 1) + δ² Σ_g w(g) = (G − 1) + O( δ² · n ),
$$ (55)

so Q → ∞ with n for any δ > 0. The *magnitude* of the spread, not its p-value, is therefore the informative readout, and Section 5.1.3 reports a random-grouping control demonstrating exactly this failure of the p-value.

#### 3.5.4 Decomposing the inflation: difficulty against familiarity

That an aggregate understates a model's error is an observation; attributing it is a separate step, and the same reference model performs it.

**Theorem 3 (Exposure-bias decomposition).** Let Ē_A^all denote A's error aggregated over all evaluation groups and Ē_A^out its error on the unexposed stratum. Define the total inflation

$$
Ι = Ē_A^out − Ē_A^all ≥ 0.
$$ (56)

Let R be a reference satisfying the condition of Proposition 2, and let

$$
Δ_R = Ē_R^out − Ē_R^all
$$ (57)

be R's corresponding shift across the identical strata. Then Ι admits the decomposition

$$
Ι = κ · Δ_R + Φ,   Φ = Ι − κ · Δ_R,   ϕ = Φ / Ι ,
$$ (58)

in which κ · Δ_R is the *intrinsic-difficulty component* — the amount by which the unexposed stratum is harder for any model — Φ is the *familiarity component* attributable to asymmetric exposure, and ϕ is the familiarity fraction. The transfer coefficient κ converts a difficulty shift measured in R's error units into A's, and **its value is not identified by the data; it is fixed by the difficulty model one adopts**:

$$
κ = 1  (additive difficulty),    κ = c_A / c_R  (multiplicative difficulty, as in (44)).
$$ (59)

*Proof.* R has no differential familiarity, so by construction the whole of Δ_R is intrinsic difficulty measured in R's error units. Under the additive model ē_M(g) = d(g) + c_M − φ_M𝟙[·], a difficulty shift transfers between models unchanged, so κ = 1. Under the multiplicative model (44) a shift scales with the model's capability constant, so κ = c_A/c_R. In either case the residual Φ is by definition the part of Ι that difficulty does not explain. ∎

Two consequences must be stated with the theorem, because the choice of κ is consequential and is a genuine limitation of the instrument rather than a presentational detail.

First, **κ = 1 is the least conservative reading, not the most.** Since Φ decreases in κ and c_A/c_R ≥ 1 whenever A is the weaker model, the additive reading *maximises* the familiarity attribution. An auditor who wants a lower bound on Φ should use the multiplicative reading.

Second, **the two readings should therefore be reported as an interval** rather than a point estimate. Writing κ_max = c_A/c_R,

$$
Φ ∈ [ Ι − κ_max Δ_R ,  Ι − Δ_R ] ,   ϕ ∈ [ 1 − κ_max Δ_R/Ι ,  1 − Δ_R/Ι ] .
$$ (60)

Section 5.1.2 reports Ι = 1.905°, Δ_R = 0.837° and κ_max = 1.769 for the affected modality, giving Φ ∈ [0.425°, 1.068°] and ϕ ∈ [22 %, 56 %]. The interval is wide, and no claim in this paper rests on its upper end. What survives the whole of it is the qualitative statement that answers the natural sceptical objection — that the held-out groups were simply harder — quantitatively rather than rhetorically: harder they were, but under neither difficulty model does difficulty account for all of the observed gap, and a familiarity component of at least 0.425° remains. This is a quantity lineage analysis cannot supply at all, since knowing *that* a model saw a corpus does not say how much of its advantage came from having seen it.

Algorithm 2 states the complete audit.

```algorithm
Algorithm 2 — GERA: Group Error-Ratio Audit for asymmetric exposure
Require: per-sample errors e_A(i), e_R(i) over a shared evaluation set ℰ; group labels γ(i);
         optional membership labels 𝟙[g ∈ 𝒯_A]; bootstrap replicates B; control grouping γ_ctl
Ensure: ratio profile r(·), separation AUC_r and p, heterogeneity (Q, I²), decomposition (Ι, Δ_R, Φ, ϕ)
 1: // Phase 1: per-group ratio profile (Definition 3)
 2: for all g ∈ unique(γ) do
 3:     ē_A(g) ← mean{ e_A(i) : γ(i) = g };  ē_R(g) ← mean{ e_R(i) : γ(i) = g }
 4:     r(g) ← ē_A(g) / ē_R(g)
 5: end for
 6: // Phase 2: separation, only if membership labels are available
 7: if membership labels available then
 8:     AUC_r ← Mann–Whitney statistic of { r(g) : g ∉ 𝒯_A } against { r(g) : g ∈ 𝒯_A }   ▷ Eq. (48)
 9:     p ← exact two-sided permutation p-value                                            ▷ Eq. (49)
10:     CI ← Hanley–McNeil interval on AUC_r                                               ▷ Eq. (50)
11:     φ̂_A ← 1 − mean_{g ∈ 𝒯_A} r(g) / mean_{g ∉ 𝒯_A} r(g)                                ▷ Eq. (47)
12: end if
13: // Phase 3: heterogeneity, computable with no membership labels at all
14: for all g do v(g) ← bootstrap variance of r(g) over B within-group resamples end for  ▷ Eq. (51)
15: w(g) ← 1 / v(g);  r̄ ← Σ w(g) r(g) / Σ w(g)                                            ▷ Eq. (52)
16: Q ← Σ_g w(g) ( r(g) − r̄ )²;  I² ← max(0, (Q − (G−1))/Q)                               ▷ Eqs. (53)–(54)
17: spread ← max_g r(g) / min_g r(g)                        ▷ report magnitude, not p (Eq. 55)
18: // Phase 4: negative control on a grouping where exposure is impossible
19: repeat lines 2–17 with γ ← γ_ctl;  record (Q_ctl, I²_ctl, spread_ctl)
20: // Phase 5: bias decomposition against the reference                                   ▷ Theorem 3
21: Ι ← Ē_A^out − Ē_A^all;  Δ_R ← Ē_R^out − Ē_R^all
22: Φ ← Ι − Δ_R · (c_A/c_R);  ϕ ← Φ / Ι
23: return r(·), AUC_r, p, CI, (Q, I², spread), (Q_ctl, I²_ctl, spread_ctl), (Ι, Δ_R, Φ, ϕ)
```

#### 3.5.5 Scope, and what the detector does not claim

Three limitations are structural rather than incidental and are stated with the method rather than deferred.

*The statistic is a screening instrument.* Separation is strong but imperfect — Section 5.1.3 reports one training group sitting at r = 2.18, well inside the unexposed range — so a single group's ratio must never be treated as proof of membership. The appropriate use is to convert an unfalsifiable suspicion into a testable one at zero additional experimental cost, and then to run the stratified decomposition of Theorem 3, which is the substantive analysis.

*A suitable reference must exist.* If every available model shares the audited model's partition, or if the candidate reference is itself differentially familiar across the evaluation groups, then by Proposition 2 the cancellation fails and r(g) confounds difficulty with familiarity. The near-flatness of a candidate reference's own profile is weak evidence in favour of its admissibility, but it is not a proof, since a reference exposed to *all* groups and a reference exposed to *none* both produce flat profiles.

*No false-positive guarantee is offered.* Unlike black-box contamination proofs [@oren] which yield a formal false-positive bound under an exchangeability assumption, GERA offers no such guarantee. Its value lies in the access regime it operates under: it needs neither model's training data, neither model's lineage documentation, and no corpus access — only the two prediction streams and the group labels, which are artefacts an external auditor already possesses.

#### 3.5.6 The exposure-disjoint comparison protocol (EDCP)

Table 5 states the procedure applied throughout the remainder of this work. Steps 1–2 are established practice [@kapoor][@leakage]. Steps 3–5 formalise, for group-structured corpora, the discipline that lineage analysis prescribes informally [@tsfm], and step 6 adds the prediction-only audit developed above.

**Table 5.** The exposure-disjoint comparison protocol, and the fault each step addresses.

| # | Step | Addresses | Formal object |
|---|---|---|---|
| 1 | Establish an explicit grouping key for every corpus | Precondition 2 of Definition 2 | γ : [n] → 𝒢 |
| 2 | Partition group-disjointly within each model's own pipeline | First-order leakage | Constraint (35), Algorithm 1 |
| 3 | Intersect the evaluation sample with *every* compared model's training partition, not only the audited model's | Precondition 3 of Definition 2 | γ(ℰ) ∩ 𝒯_M for all M |
| 4 | Restrict the comparison stratum to groups outside every compared model's training partition | Asymmetric exposure | ℰ* = { i : γ(i) ∉ ⋃_M 𝒯_M } |
| 5 | Where step 4 is impossible because a baseline trained on the whole corpus, report that baseline's figure as a **lower bound** and state the direction of the residual bias | Residual bias | Corollary 1, sign of (41) |
| 6 | Audit the residual with the per-group ratio r(g) against a reference model | Undocumented splits | Algorithm 2 |

Step 5 is not a formality in the present study. The near-flatness of the pretrained gaze baseline's per-group profile (Section 5.1.3) is itself diagnostic of its having trained on all fifteen MPIIGaze participants, so its held-out figure is a lower bound on its true cross-person error. Every comparison in this work is therefore *conservative with respect to the proposed networks* rather than favourable to them — a point worth stating plainly, because the comparison ends unfavourably for one modality and the reader is entitled to know that the bias runs the other way.

Formally, writing Ē_R^LB for the reported baseline figure and Ē_R^true for its unobserved cross-person error, step 5 asserts

$$
Ē_R^true ≥ Ē_R^LB  ⟹  Δ̂ = Ē_A − Ē_R^LB ≥ Ē_A − Ē_R^true = Δ*,
$$ (61)

so the reported contrast Δ̂ overstates rather than understates the proposed network's disadvantage only in the direction that could reverse a favourable conclusion, never one that could manufacture one.

Applying steps 3–4, all gaze results in Section 5 use the two held-out participants only. A 2,000-eye-per-participant cap inherited from the pretrained evaluation initially restricted the comparison to 4,000 samples; both networks were re-run over the full 64,172. We note that the established protocol for person-independent evaluation on MPIIGaze is leave-one-person-out over all fifteen participants [@mpiigaze_pami]; the single group-disjoint split used here is weaker, and Section 6 records this as a limitation of the present *study* rather than of the protocol in Table 5, which is agnostic to how many folds are run.
