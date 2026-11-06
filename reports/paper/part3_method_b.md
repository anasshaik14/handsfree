### 3.6 Network architectures

All four networks are defined from scratch rather than fine-tuned from an off-the-shelf backbone, for two reasons that are methodological rather than aesthetic. First, a fine-tuned backbone imports an undocumented exposure set from its pretraining corpus, which by Definition 2 is precisely the condition that makes a downstream comparison asymmetrically exposed; training from scratch on a corpus with a known grouping key keeps 𝒯_A explicit. Second, the concurrency constraint (3) admits parameter budgets two to three orders of magnitude below standard backbones, so the relevant design space is not reachable by pruning a large network.

#### 3.6.1 Common operators

Each network is a composition of a convolutional feature extractor and a fully connected head. Write the two-dimensional convolution with kernel **K** ∈ ℝ^(C_out × C_in × k × k), bias **b**, stride 1 and padding p = (k−1)/2 as

$$
( Conv_{𝐊,𝐛}(x) )_{c,u,v} = b_c + Σ_{c′=1}^{C_in} Σ_{i=1}^{k} Σ_{j=1}^{k} K_{c,c′,i,j} · x_{c′, u+i−p−1, v+j−p−1},
$$ (62)

the rectifier and max-pooling as

$$
ReLU(z) = max(0, z),   ( MaxPool_2(x) )_{c,u,v} = max_{(i,j) ∈ {0,1}²} x_{c, 2u+i, 2v+j},
$$ (63)

batch normalisation [@batchnorm] over a minibatch ℬ as

$$
BN_{γ,β}(z) = γ · ( z − μ_ℬ ) / √( σ_ℬ² + ϵ ) + β,   μ_ℬ = 𝔼_ℬ[z],  σ_ℬ² = Var_ℬ[z],
$$ (64)

adaptive average pooling to unit spatial extent as

$$
( GAP(x) )_c = (1/(HW)) Σ_{u=1}^{H} Σ_{v=1}^{W} x_{c,u,v},
$$ (65)

inverted dropout [@dropout] at rate q as

$$
Drop_q(z) = z ⊙ 𝐦 / (1 − q),   m_j ∼ Bernoulli(1 − q)  i.i.d.,
$$ (66)

and the affine layer as

$$
Lin_{𝐖,𝐛}(z) = 𝐖 z + 𝐛.
$$ (67)

Spatial extent after a padded 3 × 3 convolution is preserved and halved by each pooling stage:

$$
H^(l) = ⌊ H^(l−1) / 2 ⌋,   W^(l) = ⌊ W^(l−1) / 2 ⌋ .
$$ (68)

The receptive field after L such blocks, each contributing one 3 × 3 convolution followed by a stride-2 pool, grows as

$$
RF_L = 1 + Σ_{l=1}^{L} ( k − 1 ) · 2^{l−1} = 1 + 2 ( 2^{L} − 1 ) = 2^{L+1} − 1.
$$ (69)

For the three-block networks RF₃ = 15 pixels, which on a 36 × 60 gaze patch spans the full iris and both eye corners, and on a 64 × 64 face crop spans the interocular region — in both cases the anatomically informative extent, as the attribution analysis of Sections 5.2.3 and 5.3.3 confirms independently.

#### 3.6.2 The four networks

**Blink (binary classification).** Input x ∈ ℝ^(1×32×32); two convolutional blocks; head to two logits.

$$
h₁ = MaxPool₂( ReLU( Conv_{1 → w}(x) ) ) ∈ ℝ^{w×16×16},
$$ (70)

$$
h₂ = MaxPool₂( ReLU( Conv_{w → 2w}(h₁) ) ) ∈ ℝ^{2w×8×8},
$$ (71)

$$
f_blink(x) = Lin_{fc → 2}( Drop_q( ReLU( Lin_{128w → fc}( vec(h₂) ) ) ) ) ∈ ℝ².
$$ (72)

**Gaze (two-target regression).** Input x ∈ ℝ^(1×36×60); three convolutional blocks; spatial extent 36×60 → 18×30 → 9×15 → 4×7 by (68); head to two real outputs (pitch, yaw) in radians.

$$
h₃ = ( MaxPool₂ ∘ ReLU ∘ Conv_{2w → 4w} ) ( MaxPool₂ ∘ ReLU ∘ Conv_{w → 2w} ) ( MaxPool₂ ∘ ReLU ∘ Conv_{1 → w} ) (x) ∈ ℝ^{4w×4×7},
$$ (73)

$$
f_gaze(x) = Lin_{fc → 2}( Drop_q( ReLU( Lin_{112w → fc}( vec(h₃) ) ) ) ) ∈ ℝ².
$$ (74)

**Head pose (three-target regression).** Input x ∈ ℝ^(3×64×64); three convolutional blocks; extent 64 → 32 → 16 → 8; head to three real outputs (yaw, pitch, roll) in degrees.

$$
h₃ = ( MaxPool₂ ∘ ReLU ∘ Conv_{2w → 4w} ) ( MaxPool₂ ∘ ReLU ∘ Conv_{w → 2w} ) ( MaxPool₂ ∘ ReLU ∘ Conv_{3 → w} ) (x) ∈ ℝ^{4w×8×8},
$$ (75)

$$
f_pose(x) = Lin_{fc → 3}( Drop_q( ReLU( Lin_{256w → fc}( vec(h₃) ) ) ) ) ∈ ℝ³.
$$ (76)

**Voice (six-way classification).** Input x ∈ ℝ^(1×40×81); three convolutional blocks with batch normalisation, the last followed by global average pooling, which makes the network length-agnostic so that variable-duration utterances require no padding or truncation at inference:

$$
h₁ = MaxPool₂( ReLU( BN( Conv_{1 → w}(x) ) ) ),
h₂ = MaxPool₂( ReLU( BN( Conv_{w → 2w}(h₁) ) ) ),
$$ (77)

$$
h₃ = GAP( ReLU( BN( Conv_{2w → 4w}(h₂) ) ) ) ∈ ℝ^{4w×1×1},
$$ (78)

$$
f_voice(x) = Lin_{fc → 6}( Drop_q( ReLU( Lin_{4w → fc}( vec(h₃) ) ) ) ) ∈ ℝ⁶.
$$ (79)

Global average pooling in (78) is the structural reason the voice network is by far the smallest of the four despite having the largest input: it discards the spatial extent before the fully connected head, so the head's input dimension is 4w rather than 4w · H · W.

#### 3.6.3 Parameter and computation counts

For a padded k × k convolution the parameter count is C_out(C_in k² + 1), for batch normalisation 2C, and for an affine layer d_out(d_in + 1). The total is therefore

$$
|ϑ| = Σ_{l ∈ conv} C_out^(l) ( C_in^(l) k² + 1 ) + Σ_{l ∈ bn} 2 C^(l) + Σ_{l ∈ fc} d_out^(l) ( d_in^(l) + 1 ).
$$ (80)

Applying (80) to the four tuned configurations of Table 8 gives closed-form counts that match the deployed modules exactly:

$$
|ϑ_blink| = 160 + 4,640 + 131,136 + 130 = 136,066   (w = 16, fc = 64),
$$ (81)

$$
|ϑ_gaze| = 320 + 18,496 + 73,856 + 458,880 + 258 = 551,810   (w = 32, fc = 128),
$$ (82)

$$
|ϑ_pose| = 896 + 18,496 + 73,856 + 1,048,704 + 387 = 1,142,339   (w = 32, fc = 128),
$$ (83)

$$
|ϑ_voice| = (320 + 64) + (18,496 + 128) + (73,856 + 256) + 8,256 + 390 = 101,766   (w = 32, fc = 64).
$$ (84)

Two structural facts follow from (81)–(84) and govern the efficiency analysis of Section 5.7. First, in the three image networks the first fully connected layer dominates: it accounts for 96.4 %, 83.2 % and 91.8 % of blink, gaze and head-pose parameters respectively. Second, the voice network — alone in using global average pooling — inverts this, with 72.6 % of its parameters in the *convolutional* stack. The multiply–accumulate count runs the other way, since convolutions are applied at every spatial location:

$$
MAC_conv^(l) = C_out^(l) · C_in^(l) · k² · H_out^(l) · W_out^(l),   MAC_fc^(l) = d_out^(l) · d_in^(l),
$$ (85)

$$
MAC_total = Σ_l MAC^(l) ,   FLOPs ≈ 2 · MAC_total .
$$ (86)

A model can therefore be parameter-heavy and computation-light (the image networks) or parameter-light and computation-heavy relative to its size (the voice network), and Section 3.12 shows that at these sizes on CPU it is the *former* that binds, because inference is memory-bound rather than arithmetic-bound.

### 3.7 Training objectives

#### 3.7.1 Classification losses

Blink and voice are trained by cross-entropy over the softmax of the logits:

$$
p_c(x) = exp( f_c(x) ) / Σ_{c′} exp( f_{c′}(x) ),
$$ (87)

$$
ℓ_CE( f(x), y ) = − log p_y(x) = − f_y(x) + log Σ_{c′} exp( f_{c′}(x) ).
$$ (88)

Where class priors are markedly unequal — the blink evaluation set is 65.5 % closed — the balanced variant reweights by inverse prevalence,

$$
ℓ_CE^bal( f(x), y ) = − ( 1 / ( C · π̂_y ) ) log p_y(x),   π̂_c = n_c / n ,
$$ (89)

which is the estimator whose minimiser is Bayes-optimal under the *balanced* risk, and therefore the loss consistent with reporting balanced accuracy as a headline metric.

#### 3.7.2 Regression losses and the axis-scaling correction

Gaze is trained by mean squared error in the (pitch, yaw) plane,

$$
ℓ_gaze( f(x), θ ) = ‖ f(x) − θ ‖₂² = ( θ̂_p − θ_p )² + ( θ̂_y − θ_y )² ,
$$ (90)

with both axes in radians and therefore commensurable. Head pose is not commensurable in this way: in 300W-LP the marginal standard deviations of yaw, pitch and roll differ by more than a factor of three, so an unnormalised sum

$$
ℓ_pose^naive = Σ_{a ∈ {y,p,r}} ( θ̂_a − θ_a )²
$$ (91)

allocates gradient in proportion to each axis's *variance* rather than to its interaction relevance, and in practice lets yaw dominate the other two. The corrected objective standardises each axis by its training-set marginal standard deviation,

$$
ℓ_pose = Σ_{a ∈ {y,p,r}} ( ( θ̂_a − θ_a ) / σ_a )² ,   σ_a² = Var_{𝒯}[ θ_a ],
$$ (92)

which equalises the per-axis contribution to the gradient. Section 5.3 shows that even under (92) the two narrow-support axes are reproduced at approximately half gain while the wide-support yaw axis is not, establishing that the residual asymmetry is a property of the available target support rather than of the loss weighting.

#### 3.7.3 Optimisation

Let ϑ_t denote parameters at step t and g_t = ∇_ϑ ℓ̂(ϑ_t) the minibatch gradient. Adam [@adam] maintains exponential moving averages of the first and second gradient moments,

$$
m_t = β₁ m_{t−1} + (1 − β₁) g_t,   v_t = β₂ v_{t−1} + (1 − β₂) g_t ⊙ g_t ,
$$ (93)

$$
m̂_t = m_t / (1 − β₁^t),   v̂_t = v_t / (1 − β₂^t),
$$ (94)

$$
ϑ_{t+1} = ϑ_t − η_t · m̂_t / ( √(v̂_t) + ϵ ) ,
$$ (95)

with β₁ = 0.9, β₂ = 0.999, ϵ = 10⁻⁸. AdamW [@adamw] decouples the weight-decay term from the adaptive denominator,

$$
ϑ_{t+1} = ϑ_t − η_t · [ m̂_t / ( √(v̂_t) + ϵ ) + λ ϑ_t ] ,
$$ (96)

which is the variant selected by the search in three of four modalities (Table 8). The distinction matters here for a reason connected to Section 3.5: L2 regularisation folded into the gradient is scaled by 1/√v̂_t and therefore applied unequally across parameters, whereas the decoupled form (96) applies a uniform contraction, and uniform contraction is what suppresses the identity-specific co-adaptation that a group-disjoint split penalises.

Learning-rate schedules are either constant, step, or cosine-annealed [@sgdr]:

$$
η_t^step = η₀ · γ^{⌊ t / T_step ⌋},  γ = 0.5,  T_step = ⌊ T_max/3 ⌋,
$$ (97)

$$
η_t^cos = η_min + ½ ( η₀ − η_min ) ( 1 + cos( π t / T_max ) ).
$$ (98)

The complete training objective, including decoupled decay and augmentation, is

$$
ϑ* = arg min_ϑ  (1/|𝒯|) Σ_{i ∈ 𝒯} 𝔼_{ω ∼ p_ω} [ ℓ( f(T_ω(xᵢ) ; ϑ), 𝒯_ω(yᵢ) ) ] + (λ/2) ‖ϑ‖₂² .
$$ (99)

### 3.8 Hyperparameter search under the corrected protocol

With the protocol repaired, optimisation can be *attributed* rather than merely observed. Let the search space be the product

$$
𝒮 = 𝒪 × Λ_lr × Λ_wd × 𝒬 × 𝒲 × ℱ × 𝒮_ch × 𝒜_flag × ℬ,
$$ (100)

comprising optimiser, learning rate, weight decay, dropout rate, convolutional width, fully connected width, schedule, augmentation flag and batch size, with the two scale parameters sampled log-uniformly,

$$
log η ∼ 𝒰( log 10⁻⁴, log 3 × 10⁻³ ),   log λ ∼ 𝒰( log 10⁻⁶, log 10⁻³ ),
$$ (101)

and dropout sampled uniformly on [0.1, 0.5]. Random search [@bergstra] is preferred to grid search on the standard argument: if only d_eff ≪ d of the d dimensions materially affect the objective, a grid wastes its budget resolving irrelevant axes, whereas the probability that T random draws include at least one within the top-ε quantile of the objective is

$$
ℙ[ at least one ε-good trial ] = 1 − (1 − ε)^T ,
$$ (102)

independently of d. With T = 20 and ε = 0.10 this is 0.878; with T = 27, 0.942.

Two design decisions make the search interpretable rather than merely effective. First, **trial 0 of every search is pinned to the original deployed configuration**, so that the search is directly comparable to the deployed baseline and the reported gain is a difference between two points in the same experiment rather than across experiments. Second, **selection uses the validation partition and each test partition is read exactly once**, after selection has closed:

$$
ϑ̂ = arg min_{c ∈ 𝒞_T} R̂( f_{ϑ(c)} ; 𝒱 ),   report R̂( f_{ϑ̂} ; ℰ )  once.
$$ (103)

Initialisation is seeded, so every trial reproduces from its logged configuration row. Algorithm 3 states the procedure.

```algorithm
Algorithm 3 — Pinned-baseline random search under the corrected protocol
Require: dataset D with group labels γ; search space 𝒮; budget T; baseline config c₀; seed σ
Ensure: selected config ĉ, validation curve, single test read
 1: 𝒯, 𝒱, ℰ ← GroupDisjointPartition(γ, φ = (0.7, 0.15, 0.15), σ)          ▷ Algorithm 1
 2: best ← ∞;  ĉ ← c₀
 3: for t = 0 to T − 1 do
 4:     c ← c₀ if t = 0 else SampleConfig(𝒮, rng_σ)          ▷ trial 0 pins the deployed baseline
 5:     seed_all(σ)                                          ▷ config → result must be reproducible
 6:     ϑ ← TrainToConvergence(f_c, 𝒯, objective (99), schedule (97)/(98))
 7:     s ← R̂( f_ϑ ; 𝒱 )                                     ▷ selection score, validation only
 8:     append (c, s) to reports/tuning/<model>_trials.csv    ▷ full search is reproducible/citable
 9:     if s < best then best ← s;  ĉ ← c;  ϑ̂ ← ϑ end if
10: end for
11: // test partition is read exactly once, after selection has closed
12: return ĉ, R̂( f_ϑ̂ ; ℰ )
```

A prediction follows from Section 3.4 that the search results test. Once the split stops rewarding memorisation, the binding constraint migrates from approximation error to estimation error. Decomposing the excess risk of the selected hypothesis,

$$
R(f_ϑ̂) − R* = ( inf_{f ∈ ℱ_c} R(f) − R* )  +  ( R(f_ϑ̂) − inf_{f ∈ ℱ_c} R(f) ) ,
                    ⏟ approximation error              ⏟ estimation error
$$ (104)

a group-disjoint split increases the estimation term — the effective sample size is the group count, not the row count, by (10) — without changing the approximation term. Configurations that *suppress variance* should therefore dominate configurations that *add capacity*. Section 5.1.4 reports that augmentation was selected in four searches of four and decoupled weight decay in three of four, while the selected widths remained modest, which is the predicted degeneracy.

### 3.9 Deployed decision rules

Table 6 states, for each modality, the rule Δ_m that consumes the network output, and the error statistic that rule makes diagnostic. It is the connective tissue of the entire results section: it explains why a bias of 0.1° matters more for head pose than a variance of 12°, why raw gaze angular error overstates the deployed cursor error, and why the per-frame blink error rate is not the click error rate.

**Table 6.** What each network drives, the deployed decision rule that consumes it, and the governing error statistic.

| Modality | Drives | Deployed decision rule | Governing error statistic |
|---|---|---|---|
| Gaze | OS cursor position | 16-point per-user polynomial calibration with online recursive-least-squares correction; one-euro filter; 12 px deadzone; cursor frozen 180 ms around blinks | Residual scatter *after* an affine map, not raw angular error |
| Head pose | Discrete application and tab gestures | Yaw trigger 30° on / 12° off, pitch 22° / 9°; 650 ms dwell (≈20 frames at 30 fps); 1,200 ms cooldown | Per-axis bias, and variance after dwell averaging |
| Blink | Mouse clicks | Closure ≥ 420 ms → left, ≥ 1,000 ms → right, ≥ 1,700 ms → double, ≥ 3,000 ms → cancel; 400 ms refractory; natural blinks (100–300 ms) rejected by construction | Integrity of a ≈13-frame closure run, not per-frame accuracy |
| Voice | scroll up / scroll down / pause control / resume control / enter / escape | Argmax over six keywords on a separate audio thread | Per-class precision on the safety-critical words |

#### 3.9.1 Gaze: calibration, online correction and stabilisation

The network output is never applied to the cursor directly. A per-user calibration maps eye features and head pose to screen coordinates through a regularised second-order polynomial. With normalised iris offsets (h_x, h_y) and head angles (θ_y, θ_p), the feature vector is

$$
𝛗( h_x, h_y, θ_y, θ_p ) = [ 1,  h_x,  h_y,  θ_y/30,  θ_p/30,  h_x h_y,  h_x²,  h_y² ]ᵀ ∈ ℝ⁸,
$$ (105)

and the screen prediction is the pair of linear functionals

$$
ŝ_x = 𝐰_xᵀ 𝛗,   ŝ_y = 𝐰_yᵀ 𝛗.
$$ (106)

Including the head-angle terms is what makes the map tolerant to small head motion, the dominant failure mode of iris-only calibration. Weights are fitted from the N = 16 calibration targets by ridge regression,

$$
𝐰_• = ( 𝚽ᵀ𝚽 + μ 𝐈₈ )⁻¹ 𝚽ᵀ 𝐬_• ,   𝚽 = [ 𝛗₁, …, 𝛗_N ]ᵀ ∈ ℝ^{N×8},
$$ (107)

the ridge term being necessary because the eight features are strongly collinear at small calibration spreads. A fit whose training error exceeds 250 px or whose precision exceeds 130 px is rejected and never persisted, since predictions from such a model fling the cursor into screen corners.

Calibration then tracks drift in posture and illumination by a recursive least-squares update [@rls] applied whenever the user commits a deliberate click at a known target (t_x, t_y). With inverse-covariance **P** and forgetting factor λ_f,

$$
𝐤 = 𝐏 𝛗 / ( λ_f + 𝛗ᵀ 𝐏 𝛗 ),
$$ (108)

$$
𝐰_• ← 𝐰_• + 𝐤 ( t_• − 𝐰_•ᵀ 𝛗 ),
$$ (109)

$$
𝐏 ← ( 𝐏 − 𝐤 𝛗ᵀ 𝐏 ) / λ_f .
$$ (110)

Equations (108)–(110) are the exact sequential form of the batch least-squares solution (107), so the update can only refine the existing weights and cannot diverge from them. The forgetting factor sets an effective memory length

$$
N_eff = 1 / ( 1 − λ_f ) ,
$$ (111)

which at the deployed λ_f = 0.9995 corresponds to N_eff = 2,000 samples — long enough that a single mis-registered click cannot perturb the fit, short enough to track a posture change over a session.

The calibrated estimate is stabilised by a three-stage chain. The one-euro filter [@onceuro] is an adaptive first-order low-pass whose cut-off rises with observed speed, giving heavy smoothing at fixation and low lag during saccades. For a sample x_t at time t, with Δt the interval since the previous sample, the smoothing factor for cut-off f_c is

$$
α(f_c, Δt) = 1 / ( 1 + τ/Δt ) ,   τ = 1 / ( 2π f_c ),
$$ (112)

the speed estimate is itself low-passed at a fixed derivative cut-off f_d,

$$
ẋ_t = ( x_t − x̂_{t−1} ) / Δt,   ẋ̂_t = α(f_d, Δt) ẋ_t + ( 1 − α(f_d, Δt) ) ẋ̂_{t−1},
$$ (113)

the cut-off is made speed-adaptive,

$$
f_c^(t) = f_min + β · | ẋ̂_t | ,
$$ (114)

and the output is

$$
x̂_t = α( f_c^(t), Δt ) · x_t + ( 1 − α( f_c^(t), Δt ) ) · x̂_{t−1}.
$$ (115)

The deployed configuration uses a low minimum cut-off f_min = 0.3 Hz, which trades response speed for steadiness at fixation; Section 5.2.2 shows this setting is the correct response to the limits of agreement measured for the gaze network.

A positional deadzone of radius R then absorbs residual tremor, moving the output only when the input escapes the disc and thereafter trailing it by exactly R so that motion is continuous across the boundary:

$$
𝐨_t = 𝐨_{t−1} + ( 𝐱̂_t − 𝐨_{t−1} ) · max( 0, ‖𝐱̂_t − 𝐨_{t−1}‖ − R ) / ‖𝐱̂_t − 𝐨_{t−1}‖ ,
$$ (116)

and a slew limiter caps output speed so that a single-frame estimation glitch cannot teleport the cursor:

$$
𝐜_t = 𝐜_{t−1} + ( 𝐨_t − 𝐜_{t−1} ) · min( 1, v_max Δt / ‖𝐨_t − 𝐜_{t−1}‖ ) .
$$ (117)

Algorithm 4 states the composed chain.

```algorithm
Algorithm 4 — Gaze stabilisation and online recalibration (per frame)
Require: raw features (h_x, h_y), head angles (θ_y, θ_p), time t; weights 𝐰_x, 𝐰_y, covariance 𝐏;
         one-euro parameters (f_min, β, f_d); deadzone radius R; max speed v_max; forgetting λ_f
Ensure: cursor position 𝐜_t
 1: 𝛗 ← [1, h_x, h_y, θ_y/30, θ_p/30, h_x h_y, h_x², h_y²]ᵀ                        ▷ Eq. (105)
 2: (s_x, s_y) ← (𝐰_xᵀ𝛗, 𝐰_yᵀ𝛗)                                                    ▷ Eq. (106)
 3: if blink FSM reports a closure in progress then
 4:     return 𝐜_{t−1}                                     ▷ cursor frozen ±180 ms around a blink
 5: end if
 6: Δt ← max(t − t_prev, 10⁻⁶)
 7: for each axis a ∈ {x, y} do                                                    ▷ Eqs. (112)–(115)
 8:     ẋ ← (s_a − x̂_a) / Δt;   ẋ̂_a ← α(f_d, Δt)·ẋ + (1 − α(f_d, Δt))·ẋ̂_a
 9:     f_c ← f_min + β·|ẋ̂_a|;   x̂_a ← α(f_c, Δt)·s_a + (1 − α(f_c, Δt))·x̂_a
10: end for
11: 𝐨_t ← Deadzone_R(𝐱̂_t, 𝐨_{t−1})                                                 ▷ Eq. (116)
12: 𝐜_t ← SlewLimit_{v_max}(𝐨_t, 𝐜_{t−1}, Δt)                                       ▷ Eq. (117)
13: if a deliberate click was committed at known target (t_x, t_y) then            ▷ Eqs. (108)–(110)
14:     𝐤 ← 𝐏𝛗 / (λ_f + 𝛗ᵀ𝐏𝛗)
15:     𝐰_x ← 𝐰_x + 𝐤(t_x − 𝐰_xᵀ𝛗);   𝐰_y ← 𝐰_y + 𝐤(t_y − 𝐰_yᵀ𝛗)
16:     𝐏 ← (𝐏 − 𝐤𝛗ᵀ𝐏) / λ_f
17: end if
18: return 𝐜_t
```

The structure of Algorithm 4 anticipates a result. Calibration (106) restores angular scale, which as Section 5.2.2 shows necessarily amplifies the representational noise that shrinkage was suppressing; the filter chain (112)–(117) then pays for that restoration in bandwidth. The pipeline is thus already organised around a trade the analysis will make explicit.

#### 3.9.2 Blink: duration coding and run integrity

The blink rule maps a *sustained* closure to an action. Let b_t ∈ {0,1} be the per-frame closed indicator produced by thresholding the network's closed-class probability at ζ, and let a closure run begin at t₀ = min{ t : b_t = 1 after an open state }. On reopening, the run duration D = t − t₀ selects the action by the staircase

$$
Δ_blink(D) = max { κ ∈ 𝒦 : D ≥ T_κ } ,   𝒦 = ( click, right, double, cancel ),
$$ (118)

$$
(T_click, T_right, T_double, T_cancel) = (420, 1000, 1700, 3000) ms,
$$ (119)

with any run shorter than T_click classified as a natural blink and discarded, and a refractory interval T_ref = 400 ms suppressing repeat fires after any non-natural action.

The evaluation consequence of (118) is the central point of Section 5.5 and is worth deriving. At frame rate F = 30 Hz the shortest actionable run spans

$$
n_run = ⌈ F · T_click ⌉ = ⌈ 30 × 0.420 ⌉ = 13  consecutive frames.
$$ (120)

Model per-frame errors as Bernoulli with false-positive rate FPR and false-negative rate FNR. If consecutive-frame errors were independent, the probability that noise alone manufactures a spurious click from an open eye is

$$
ℙ[ spurious run ] = FPR^{n_run},
$$ (121)

which at the measured FPR = 0.0779 is 0.0779^13 ≈ 3.9 × 10⁻¹⁵ — effectively zero. By contrast, the probability that a *genuine* 13-frame closure survives intact is

$$
ℙ[ run survives ] = ( 1 − FNR )^{n_run},
$$ (122)

which at FNR = 0.0556 is 0.9444^13 ≈ 0.475: under independence, more than half of deliberate clicks would be broken by a single dropped frame. The asymmetry between (121) and (122) is stark and runs *opposite* to the frame-level intuition that a false click is worse than a missed blink. Under the deployed rule the false negative is the damaging error, and Section 5.5 accordingly identifies thresholds *below* the default ζ = 0.5 as the defensible choice.

Independence is of course not satisfied: consecutive frames are strongly correlated, so (122) is a lower bound on run survival and (121) an upper bound on spurious firing. Modelling the error indicator as a stationary first-order Markov chain with persistence ϱ, the effective number of independent trials in a run of length n is

$$
n_eff^run = n · ( 1 − ϱ ) / ( 1 + ϱ ) ,
$$ (123)

and substituting n_eff^run for n_run in (121)–(122) preserves the direction of the asymmetry while moderating its magnitude. Section 6 records the measurement of ϱ in live sessions as an open item, since it is not identifiable from the per-frame corpus.

```algorithm
Algorithm 5 — Intentional-blink finite-state machine
Require: per-frame closed probability p_t; threshold ζ; stage thresholds T_click < T_right < T_double < T_cancel;
         refractory T_ref
Ensure: click events; live progress readout for user feedback
 1: state ← OPEN;  t₀ ← 0;  t_ref ← −∞
 2: for each frame at time t do
 3:     b ← 𝟙[ p_t ≥ ζ ]
 4:     if state = REFRACTORY then
 5:         if t ≥ t_ref then state ← OPEN else continue end if
 6:     end if
 7:     if state = OPEN and b = 1 then state ← CLOSED;  t₀ ← t;  continue end if
 8:     if state = CLOSED then
 9:         D ← (t − t₀)·1000                                             ▷ closure duration, ms
10:         κ ← max{ κ′ ∈ 𝒦 : D ≥ T_{κ′} }  (natural if none)              ▷ Eq. (118)
11:         emit progress(D, κ)                                           ▷ UI + audio stage feedback
12:         if b = 0 then                                                 ▷ reopened: commit
13:             emit event(κ, D)
14:             if κ ≠ natural then t_ref ← t + T_ref;  state ← REFRACTORY
15:             else state ← OPEN end if
16:         end if
17:     end if
18: end for
```

#### 3.9.3 Head pose: hysteresis, dwell and the gain correction

A gesture fires only when an axis crosses its ON threshold and *remains* across it for the dwell time, and nothing further fires until the axis returns inside the OFF threshold and the cooldown elapses. For axis a with smoothed angle θ̃_a, the trigger predicate is

$$
Fire_a(t) = 𝟙[ |θ̃_a(t′)| ≥ θ_on^a  ∀ t′ ∈ [t − T_dwell, t] ] · 𝟙[ t − t_last^a ≥ T_cd ] ,
$$ (124)

with release when |θ̃_a| ≤ θ_off^a, and deployed parameters

$$
(θ_on^yaw, θ_off^yaw) = (30°, 12°),  (θ_on^pitch, θ_off^pitch) = (22°, 9°),  T_dwell = 650 ms,  T_cd = 1200 ms.
$$ (125)

The raw estimate is pre-smoothed by an exponential moving average with α_g = 0.35 so that a single-frame estimation spike cannot satisfy the dwell predicate:

$$
θ̃_a(t) = α_g θ_a(t) + ( 1 − α_g ) θ̃_a(t−1).
$$ (126)

Axis arbitration selects whichever axis has the greater headroom relative to its own threshold, preventing diagonal double-fires:

$$
a* = arg max_{a ∈ {yaw, pitch}} | θ̃_a | / θ_on^a ,
$$ (127)

with the non-dominant axis held in reset whenever its ratio falls below 0.6.

Two quantitative consequences follow and are tested in Section 5.3. First, the dwell requirement averages

$$
n_dwell = ⌈ F · T_dwell ⌉ = ⌈ 30 × 0.650 ⌉ = 20  frames,
$$ (128)

and for a zero-mean residual with per-frame standard deviation σ and AR(1) correlation ϱ the standard deviation of the dwell mean is

$$
σ_dwell = ( σ / √n_dwell ) · √( ( 1 + ϱ ) / ( 1 − ϱ ) ) ,
$$ (129)

which at ϱ = 0 reduces to σ/√20 ≈ 0.224 σ. Crucially, (129) reduces *variance* but leaves *bias* untouched: a systematic offset survives dwell averaging entirely. This inverts the usual ranking of the two error components and is why Table 6 names per-axis bias as the governing statistic for this modality.

Second, if the network reproduces true angle at fitted gain s < 1 — the shrinkage phenomenon of Section 5.2.2 — then a trigger nominally at θ_on is crossed only when the *true* angle reaches

$$
θ_true^trigger = θ_on / s .
$$ (130)

At the measured yaw gain s = 0.822 this is 30°/0.822 ≈ 36.5°. The offset is systematic, stable and learnable rather than erratic, which is precisely why the gesture layer remains usable at an 8.74° yaw mean absolute error.

```algorithm
Algorithm 6 — Deliberate head-gesture finite-state machine (per axis, with arbitration)
Require: raw angles θ_yaw, θ_pitch at time t; thresholds (θ_on^a, θ_off^a); dwell T_dwell; cooldown T_cd;
         EMA factor α_g; optional repeat interval T_rep
Ensure: gesture events {head_left, head_right, head_up, head_down}
 1: θ̃_a ← α_g θ_a + (1 − α_g) θ̃_a  for a ∈ {yaw, pitch}                            ▷ Eq. (126)
 2: a* ← arg max_a |θ̃_a| / θ_on^a;   if |θ̃_{ā}|/θ_on^{ā} < 0.6 then reset(ā) end if  ▷ Eq. (127)
 3: d ← +1 if θ̃_{a*} ≥ θ_on^{a*};  −1 if θ̃_{a*} ≤ −θ_on^{a*};  0 otherwise
 4: switch state(a*)
 5:   case IDLE:
 6:       if d ≠ 0 and (t − t_last) ≥ T_cd then state ← PENDING; dir ← d; t_enter ← t end if
 7:   case PENDING:
 8:       if d ≠ dir then state ← IDLE                       ▷ left the band: not deliberate
 9:       else if (t − t_enter) ≥ T_dwell then
10:           state ← FIRED;  t_last ← t;  emit gesture(a*, dir)                    ▷ Eq. (124)
11:       end if
12:   case FIRED:
13:       if |θ̃_{a*}| ≤ θ_off^{a*} then state ← IDLE          ▷ return-to-neutral hysteresis
14:       else if repeat and d = dir and (t − t_last) ≥ T_rep then
15:           t_last ← t;  emit gesture(a*, dir)
16:       end if
17: end switch
```

#### 3.9.4 Voice: argmax with an optional reject option

The deployed voice rule is forced-choice argmax over the six keyword posteriors,

$$
Δ_voice(x) = arg max_{c ∈ 𝒞} p_c(x),
$$ (131)

executed on a dedicated audio thread. Section 5.4 evaluates the alternative *selective* rule with a reject option,

$$
Δ_voice^ζ(x) = arg max_c p_c(x)  if  max_c p_c(x) ≥ ζ ;  ⊥ (abstain)  otherwise,
$$ (132)

whose behaviour is summarised by the coverage and selective-risk pair

$$
cov(ζ) = ℙ[ max_c p_c(x) ≥ ζ ],   risk(ζ) = ℙ[ Δ_voice(x) ≠ y  |  max_c p_c(x) ≥ ζ ] ,
$$ (133)

and the risk–coverage curve ζ ↦ (cov(ζ), risk(ζ)). The comparison against a production ASR engine equipped with a native rejection option is only meaningful on this curve, since a single accuracy figure cannot express the trade the two systems make differently. In an interface a substitution executes an unintended command whereas an abstention costs one repetition, so the operational loss is asymmetric,

$$
L = c_sub · ℙ[ substitution ] + c_abs · ℙ[ abstention ],   c_sub ≫ c_abs ,
$$ (134)

and the optimal threshold under (134) is the Chow rule [@chow][@geifman] ζ* = 1 − c_abs/c_sub.

#### 3.9.5 Arbitration and safety interlocks

Concurrent modalities can issue conflicting actions, and three interlocks resolve them. The cursor is frozen for ±180 ms around any detected closure, since eyelid motion perturbs the iris estimate exactly when a click is being committed; physical mouse movement suspends gaze control for 2 s, giving the user unconditional override; and the voice commands *pause control* and *resume control* gate the entire perception stack. Writing 𝟙_pause for the gate state, the emitted action set is

$$
𝒮(𝐈ₜ, 𝐚ₜ) = 𝟙[ ¬pause ] · ⋃_{m} Δ_m( f_m( 𝒞_m(𝐈ₜ, 𝐚ₜ) ) ) ∪ { resume if voice = "go" } .
$$ (135)

Equation (135) contains a design fault that Section 5.4 identifies quantitatively and that no amount of model improvement would repair: the *resume* command is the only route back once the interface has been paused, so its recognition must be robust precisely when everything else is suppressed. Section 5.4 shows the keyword bound to it is the weakest of the six, and that rebinding it is a zero-cost remedy.

### 3.10 Attribution on the deployed modules

Accuracy establishes that a network is right; it does not establish that it is right for a *transferable* reason, which for a system whose users are unseen is the property that actually matters. Attribution is computed by Integrated Gradients [@ig] against a zero baseline x′ = 0 over M = 32 steps, directly on the deployed TorchScript modules, so that what is explained is what executes.

For a scalar output F and input x, the attribution of input coordinate j is the path integral

$$
IG_j(x) = ( x_j − x′_j ) ∫_{0}^{1} ∂F( x′ + a(x − x′) ) / ∂x_j  da ,
$$ (136)

approximated by the Riemann sum

$$
IĜ_j(x) = ( x_j − x′_j ) · (1/M) Σ_{m=1}^{M} ∂F( x′ + (m/M)(x − x′) ) / ∂x_j .
$$ (137)

Integrated Gradients satisfies *completeness*,

$$
Σ_j IG_j(x) = F(x) − F(x′),
$$ (138)

which is what converts an attribution plot from an illustration into an audit: the decomposition is exact rather than indicative, so a waterfall chart of region contributions genuinely sums to the model's output displacement. It further satisfies sensitivity (a coordinate that changes the output receives non-zero attribution), implementation invariance (functionally identical networks receive identical attributions), and linearity, and is the Aumann–Shapley [@aumann] continuous analogue of the Shapley value [@shapley], inheriting its cooperative-game justification for the allocation.

Because these networks consume images and spectrograms rather than named columns, the input is partitioned into a regular 4 × 4 grid ℛ = {R₁, …, R₁₆} and each region assumes the role of a feature:

$$
Attr(R_k ; x) = Σ_{j ∈ R_k} IĜ_j(x),   Σ_{k=1}^{16} Attr(R_k ; x) = F(x) − F(x′).
$$ (139)

Four views are reported for every modality. The **global summary** ranks regions by mean absolute attribution, 𝔼_x | Attr(R_k ; x) |, with the per-sample distribution for each. The **waterfall** shows the cumulative region contributions carrying two extreme instances from baseline output to prediction. The **similarity-ordered** global explanation stacks per-instance region attributions with instances sorted by resemblance of their explanation vectors, which exposes whether the network applies one continuously varying rule or a small number of discrete strategies. The **dataset-ordered** global explanation stacks the same attributions in acquisition order; where the corpus is ordered by subject — as MRL is — stationarity under this ordering is a direct test of subject-invariance of the decision mechanism, and is used as such in Section 5.5.

```algorithm
Algorithm 7 — Region-level Integrated-Gradients attribution on a deployed module
Require: TorchScript module F; input x; baseline x′ = 0; steps M = 32; region partition ℛ (4×4 grid)
Ensure: region attributions Attr(R_k; x) with Σ_k Attr(R_k; x) = F(x) − F(x′)
 1: G ← 0                                                       ▷ accumulator, shape of x
 2: for m = 1 to M do
 3:     x^(m) ← x′ + (m/M)·(x − x′)                              ▷ point on the straight-line path
 4:     G ← G + ∇_x F(x^(m))                                     ▷ one backward pass per step
 5: end for
 6: IĜ ← (x − x′) ⊙ G / M                                        ▷ Eq. (137)
 7: for k = 1 to |ℛ| do Attr(R_k) ← Σ_{j ∈ R_k} IĜ_j end for      ▷ Eq. (139)
 8: assert | Σ_k Attr(R_k) − (F(x) − F(x′)) | < δ_tol            ▷ completeness check, Eq. (138)
 9: return Attr(·)
```

### 3.11 Evaluation protocols and metrics

#### 3.11.1 Protocols

Three protocols are used and are always named where a figure is quoted. The **published** protocol is the original random-row split, reported only in Section 5.1 to quantify what the correction was worth. The **group-disjoint** protocol applies Algorithm 1 and constraint (35), and is used for every reported score thereafter. The **exposure-disjoint** protocol additionally applies EDCP steps 3–4, restricting the comparison stratum to groups outside every compared model's training partition, and is used for every cross-model comparison.

All comparisons are per-sample matched: both models score the *identical* evaluation samples, each receiving its own native preprocessing, and ground-truth alignment between the two prediction streams is verified numerically before scoring. This removes metric-definition drift and sample-selection drift as competing explanations for any measured difference.

#### 3.11.2 Classification metrics

With confusion counts TP, TN, FP, FN and C classes,

$$
Acc = ( TP + TN ) / ( TP + TN + FP + FN ),
$$ (140)

$$
BalAcc = ½ ( TP/(TP+FN) + TN/(TN+FP) ),
$$ (141)

$$
Prec = TP/(TP+FP),  Rec = TP/(TP+FN),  F₁ = 2·Prec·Rec/(Prec+Rec),
$$ (142)

$$
macro-F₁ = (1/C) Σ_{c=1}^{C} F₁^(c),
$$ (143)

$$
MCC = ( TP·TN − FP·FN ) / √( (TP+FP)(TP+FN)(TN+FP)(TN+FN) ),
$$ (144)

$$
κ = ( p_o − p_e ) / ( 1 − p_e ),   p_o = Acc,  p_e = Σ_c ( n_{c·} n_{·c} ) / n² .
$$ (145)

Balanced accuracy (141) and MCC (144) [@matthews] are reported alongside accuracy wherever the class prior is unequal, because (140) is prevalence-weighted and can be high for a degenerate classifier; MCC is the correlation coefficient between predicted and true binary labels and is invariant to which class is designated positive.

Threshold-free ranking quality uses the areas under the receiver-operating and precision–recall curves. The former has the probabilistic identity

$$
ROC-AUC = ℙ[ score(X⁺) > score(X⁻) ] ,
$$ (146)

which is exactly what makes the "blink galaxy" visualisation of Section 5.5 a direct visual estimate of 1 − AUC, and the latter [@davis] is

$$
PR-AUC = ∫₀¹ Prec( Rec ) d Rec ,
$$ (147)

which on a majority-positive problem exceeds ROC-AUC and is the more informative of the two under a shifted deployment prior.

Probabilistic quality is assessed by the Brier score [@brier], the logarithmic loss and the expected calibration error [@guo] over B equal-width confidence bins:

$$
Brier = (1/n) Σ_{i=1}^{n} ( p̂_i − y_i )² ,
$$ (148)

$$
LogLoss = − (1/n) Σ_{i=1}^{n} log p̂_{y_i}(x_i) ,
$$ (149)

$$
ECE = Σ_{b=1}^{B} ( |ℬ_b| / n ) · | acc(ℬ_b) − conf(ℬ_b) | ,   B = 15.
$$ (150)

Calibration matters here beyond convention: the blink state machine reports fractional progress toward each click stage to the user, which requires per-frame scores to be commensurable *across* frames rather than merely correctly ordered *within* one.

#### 3.11.3 Regression metrics

$$
MAE = (1/n) Σ_i | θ̂_i − θ_i | ,   RMSE = √( (1/n) Σ_i ( θ̂_i − θ_i )² ) ,
$$ (151)

with angular error as defined in (20) reported for gaze, and per-axis signed error

$$
bias_a = (1/n) Σ_i ( θ̂_{a,i} − θ_{a,i} ) ,   spread_a = √( Var_i[ θ̂_{a,i} − θ_{a,i} ] )
$$ (152)

reported for head pose, since (152) separates exactly the two components that dwell averaging treats differently by (129). Explained variance per axis is

$$
R²_a = 1 − Σ_i ( θ̂_{a,i} − θ_{a,i} )² / Σ_i ( θ_{a,i} − θ̄_a )² .
$$ (153)

#### 3.11.4 The shrinkage gain, and why it is the organising statistic

For each axis we fit the ordinary least-squares regression of predicted on true target,

$$
θ̂_a = s_a θ_a + c_a + ζ_a ,   ŝ_a = Cov( θ̂_a, θ_a ) / Var( θ_a ) ,   ĉ_a = 𝔼[θ̂_a] − ŝ_a 𝔼[θ_a] ,
$$ (154)

and report ŝ_a with its standard error. The parameter ŝ_a is not a curiosity; it is the organising quantity of the entire results section, for the following reason. An MSE-trained estimator facing irreducible uncertainty in its own representation is *optimally* biased toward the conditional mean. Modelling the network's internal estimate as a noisy observation of the target, z = θ + ε with ε ∼ (0, σ_ε²) independent of θ ∼ (0, σ_θ²), the minimum-MSE affine predictor is the shrinkage estimator [@stein][@jamesstein]

$$
θ̂ = s* z ,   s* = σ_θ² / ( σ_θ² + σ_ε² ) = 1 / ( 1 + σ_ε²/σ_θ² ) ∈ (0,1) ,
$$ (155)

with resulting risk

$$
MSE(s) = s² σ_ε² + (1 − s)² σ_θ² ,   MSE(s*) = σ_θ² σ_ε² / ( σ_θ² + σ_ε² ) = s* σ_ε² ,
$$ (156)

which decomposes the error at any gain into a variance term and a systematic term. Two consequences follow immediately and are tested directly in Section 5.

First, the *systematic* component of error at gain s is (1 − s)|θ|, a function of target magnitude alone with **no directional dependence**. For a two-dimensional target this predicts an error field that grows radially and is isotropic in azimuth — a strong, falsifiable prediction, since any geometric fault in cropping or normalisation would necessarily break azimuthal symmetry. Section 5.2.2 tests it and it holds.

Second, "correcting" the contraction by inverting the fitted gain must *increase* error whenever the network sits near s*, since by (156)

$$
MSE(1) − MSE(s*) = σ_ε² − s* σ_ε² = ( 1 − s* ) σ_ε² > 0 .
$$ (157)

Section 5.2.2 performs exactly this inversion — an oracle correction fitted on the evaluation data itself, hence an upper bound on any achievable recalibration — and measures a 28.5 % *degradation*, confirming that the contraction is the estimator's rational response to its own uncertainty rather than a defect to be repaired. This single measurement is what redirects the engineering effort from calibration to representation, and it is why Section 5.2.4's identity-count ceiling is decisive rather than merely interesting.

Given ŝ from (154) one may also invert (155) to recover the implied representational noise,

$$
σ̂_ε² = σ_θ² ( 1 − ŝ ) / ŝ ,
$$ (158)

which converts a regression slope into an estimate of the irreducible uncertainty in the network's internal representation and makes gains comparable across axes with different target support.

#### 3.11.5 Paired comparison and significance

All model comparisons are paired at the sample level. For per-sample errors e_A(i), e_R(i) and differences dᵢ = e_A(i) − e_R(i),

$$
d̄ = (1/n) Σ_i dᵢ ,   CI_{95%} = d̄ ± 1.96 · s_d / √n ,   s_d² = (1/(n−1)) Σ_i ( dᵢ − d̄ )² ,
$$ (159)

$$
Cohen's d = d̄ / s_d ,
$$ (160)

following the standard convention for standardised paired effect size [@cohend].

with the Wilcoxon signed-rank statistic [@wilcoxon] used in preference to the paired t-test because the error distributions are strongly right-skewed:

$$
W = Σ_{i : dᵢ ≠ 0} sgn(dᵢ) · rank( |dᵢ| ).
$$ (161)

The win rate ℙ[ e_A < e_R ] is reported alongside d̄, since a large mean difference is compatible with a substantial win rate and the two answer different questions.

#### 3.11.6 Complementarity and the fusion bound

Whether two models are worth running together is decided by their error *correlation*, not their individual accuracies. We report the Pearson and Spearman correlations of per-sample errors,

$$
r_AR = Cov( e_A, e_R ) / ( σ_{e_A} σ_{e_R} ) ,   ρ_{AR} = r_AR( rank(e_A), rank(e_R) ) ,
$$ (162)

and the *oracle* per-sample selection bound

$$
Ē_oracle = (1/n) Σ_i min( e_A(i), e_R(i) ) ≤ min( Ē_A, Ē_R ) ,
$$ (163)

with the achievable fusion gain bounded above by

$$
Gain_max = 1 − Ē_oracle / min( Ē_A, Ē_R ) .
$$ (164)

Since (163) is fitted on the evaluation data — it presumes an oracle that knows which model is right per sample — it is an *upper* bound and is used only to close design questions negatively. Section 5.6 reports Gain_max of 8.8 % and 7.5 % for the two regression modalities, which is what removes fusion as an escape route from the recommendation of Section 5.7.

### 3.12 Complexity and the deployment cost model

Table 7 states the asymptotic and measured cost of each component. The measured latencies are single-thread CPU timings of the deployed TorchScript modules, not research-time proxies.

**Table 7.** Complexity of the MAPLE components. n = evaluation samples, G = groups, B = bootstrap replicates, M = IG steps, L = layers, k = kernel size, HW = spatial extent, P = |ϑ| = parameter count.

| Component | Time complexity | Space complexity | Measured (deployed) |
|---|---|---|---|
| Group-disjoint partition (Alg. 1) | O(n + G log G) | O(n + G) | < 1 s, all corpora |
| GERA audit (Alg. 2) | O(n + B·G + G²) | O(n + G) | < 2 s at G = 15, B = 1000 |
| — separation AUC (Eq. 48) | O(G_in · G_out) | O(G) | negligible |
| — heterogeneity (Eqs. 51–54) | O(B · n) | O(G) | dominated by bootstrap |
| Random search (Alg. 3) | O(T · E · n_𝒯 · C_fwd) | O(P + batch) | 10–27 trials per modality |
| Blink CNN forward | O(Σ_l C_in C_out k² H W) | O(P) = 136,066 | **0.42 ms / frame** |
| Gaze CNN forward | ″ | O(P) = 551,810 | **2.13 ms / frame** |
| Head-pose CNN forward | ″ | O(P) = 1,142,339 | **4.13 ms / frame** |
| Voice CNN forward | ″ | O(P) = 101,766 | **2.7 ms / clip** |
| One-euro + deadzone + slew (Alg. 4) | O(1) per frame | O(1) | < 0.01 ms |
| RLS calibration update (Eqs. 107–109) | O(d²) = O(64) | O(d²) | < 0.01 ms, on click only |
| Blink FSM (Alg. 5) | O(1) per frame | O(1) | negligible |
| Gesture FSM (Alg. 6) | O(1) per frame | O(1) | negligible |
| IG attribution (Alg. 7) | O(M · C_fwd) | O(P + HW) | offline only |

The deployment budget is then the instantiation of (3):

$$
τ_vis = τ(ℓ) + τ_blink + τ_gaze + τ_pose + τ_Δ = τ(ℓ) + 0.42 + 2.13 + 4.13 + τ_Δ  ms ,
$$ (165)

so the three networks contribute 6.68 ms of the 33.3 ms available at 30 fps, leaving substantial headroom for the shared landmark inference, the two state machines and the stabilisation chain. Total resident footprint is

$$
S_total = 0.57 + 2.23 + 4.60 + 0.44 = 7.84  MB ,
$$ (166)

which matters disproportionately on CPU: at these sizes inference is bound by memory traffic rather than arithmetic, and a working set that stays resident in last-level cache avoids the main-memory round trips that would otherwise dominate. The arithmetic intensity of a layer,

$$
AI^(l) = MAC^(l) / ( bytes moved^(l) ) ≈ ( C_in C_out k² H W ) / ( 4 ( C_in C_out k² + C_in H W + C_out H W ) ) ,
$$ (167)

is low for the fully connected layers that dominate the parameter counts (81)–(83), which is exactly the regime in which footprint rather than FLOPs predicts latency — and is the reason the head-pose network, with 8.4× the parameters of the blink network, costs 9.8× its latency despite a similar operator sequence.
