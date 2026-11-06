### 5.3 Head pose — discrete gesture control

Head pose does not position anything. It fires discrete application and tab gestures through the thresholded state machine of Algorithm 6: yaw triggers at 30° and releases at 12°, pitch at 22° and 9°, with a 650 ms dwell (≈20 frames at the configured 30 fps by (128)) and a 1,200 ms cooldown. Two consequences follow immediately from (124)–(129) and govern how the figures below should be read. Absolute mean absolute error is not the governing statistic — what matters is whether error can carry a pose across a trigger boundary. And because the dwell requirement averages roughly twenty frames, **bias survives averaging while variance does not**, by (129), which inverts the usual ranking of the two.

Against the 6-D rotation-representation baseline on all 1,969 AFLW2000 faces, both models scoring the identical crop with no preprocessing asymmetry, the proposed network attains 7.367° MAE (median 4.185°) against 3.482° (median 1.919°). The paired statistics are

$$
d̄ = 3.884° ± 0.272°  (95 %) ,   Cohen's d = 0.632 ,   Wilcoxon p = 1.2 × 10⁻²²⁶ ,
$$ (223)

$$
ℙ[ e_A < e_R ] = 14.58 % ,   r_AR = 0.671 ,   Ē_oracle = 3.222° ,   Gain_max = 7.5 % .
$$ (224)

The error correlation of 0.671 is *higher* than gaze's 0.491, and the oracle bound correspondingly tighter. Fusion is again unjustified, for the same structural reason: the two models fail on the same faces.

**The gain measurement transfers, with an instructive asymmetry.** Applying (154) axis-wise,

$$
ŝ_yaw^A = 0.822 ,   ŝ_pitch^A = 0.455 ,   ŝ_roll^A = 0.490 ,
$$ (225)

$$
ŝ_yaw^R = 1.046 ,   ŝ_pitch^R = 0.768 ,   ŝ_roll^R = 0.877 .
$$ (226)

Yaw — by far the widest-support axis in AFLW2000 — is the only one reproduced at near-correct scale, and the pattern is exactly what the shrinkage account (155) predicts. Since s* = σ_θ²/(σ_θ² + σ_ε²), an axis with wide target support has large σ_θ² and therefore a gain close to unity at fixed representational noise:

$$
∂ s* / ∂ σ_θ² = σ_ε² / ( σ_θ² + σ_ε² )² > 0 ,
$$ (227)

so wide target support supplies the gradient signal needed to resist contraction while the two narrow-support axes collapse to approximately half gain — the same regime as gaze. Inverting (158) per axis gives implied noise ratios

$$
σ̂_ε²/σ_θ²:  0.217  (yaw) ,  1.198  (pitch) ,  1.041  (roll) ,
$$ (228)

confirming that the representation is markedly more informative about yaw than about the other two. That yaw is the best-scaled axis is fortunate for the interface, since yaw carries the two most-used gestures.

Fig. 19 begins with the per-axis error structure, which Table 6 identifies as the decisive one.

!FIG headpose/H1_signed_violins.png | Fig. 19. Signed-error violins — per-axis bias and spread, proposed network left of each spine and 6-D rotation baseline right.

All three axes are essentially unbiased. By (152),

$$
bias^A = ( +0.109° , −0.327° , +0.062° ) ,   bias^R = ( −0.240° , −0.624° , +0.250° ) ,
$$ (229)

so the entire deficit is spread, not offset — the opposite decomposition to gaze, where a 1.987° pitch bias was the largest correctable term. This is the single most favourable result for the deployed gesture rule. Substituting the measured yaw spread into (129) at the independence limit ϱ = 0,

$$
σ_dwell^yaw = 12.49° / √20 = 2.79° ,
$$ (230)

comfortably inside the 18° hysteresis band between the 30° trigger and the 12° release. Consecutive frames are correlated, so the true reduction is smaller: at a plausible ϱ = 0.5 the same computation gives

$$
σ_dwell^yaw = ( 12.49/√20 ) · √( 1.5/0.5 ) = 2.79 × 1.732 = 4.84° ,
$$ (231)

still well inside the band. Equation (230) is therefore an upper bound on the benefit and (231) a more realistic figure; the direction is unambiguous in either case, and it holds **only because the bias is near zero**. Had the network carried a gaze-like 2° offset, no amount of dwell averaging would have removed it, since (129) attenuates only the variance term.

Fig. 20 aggregates four error measures across three axes.

!FIG headpose/H2_fingerprint.png | Fig. 20. Error fingerprint — heat grid of four error measures across yaw, pitch and roll for each model.

The fingerprint contains an axis-ranking inversion that the gain analysis resolves. The proposed network's MAE ordering is

$$
yaw (8.742°) > pitch (7.023°) > roll (6.335°) ,
$$ (232)

while the baseline's is

$$
pitch (4.492°) > yaw (3.039°) > roll (2.916°) .
$$ (233)

Yaw is simultaneously the proposed network's best-explained axis (R² = 0.875) and its worst in absolute degrees. The apparent contradiction dissolves once target ranges are considered, because the systematic error contributed by a gain deficit is (1 − s)·𝔼|θ| by (194), and the two factors move oppositely across axes:

$$
( 1 − ŝ_yaw ) · 𝔼|θ_yaw| = 0.178 × 𝔼|θ_yaw|
against
( 1 − ŝ_pitch ) · 𝔼|θ_pitch| = 0.545 × 𝔼|θ_pitch| ,
$$ (234)

and since 𝔼|θ_yaw| exceeds 𝔼|θ_pitch| by more than a factor of three in AFLW2000, an 18 % gain deficit over a ±90° range costs more degrees than a 55 % deficit over a range a fraction as wide. The inversion is therefore a *units artefact* of comparing variance-normalised against absolute measures on axes with unequal support, and quoting either alone would mislead. For gesture triggering the absolute figure is the relevant one, which is why Table 6 names it.

Fig. 21 maps error onto the pose manifold.

!FIG headpose/H3_pose_heatmap.png | Fig. 21. Pose-sphere heat map — mean error located in the yaw × pitch plane for each model.

Three-axis mean error across four yaw bands is

$$
A:  4.77°  (|yaw| < 15°, n = 891) → 5.09°  (15–30°, n = 420) → 8.08°  (30–60°, n = 380) → 18.16°  (> 60°, n = 278) ,
$$ (235)

$$
R:  2.35° → 2.47° → 3.81° → 8.20° ,
$$ (236)

a 3.8× degradation for the proposed network against 3.5× for the baseline. The proposed-to-baseline ratio moves only from

$$
4.77/2.35 = 2.03   to   18.16/8.20 = 2.21
$$ (237)

across the entire manifold. Near-parallel degradation at a near-constant ratio is strong evidence against a localised failure mode: it is the manifold-level expression of (211), an approximately constant multiplicative loss of angular resolution everywhere. Critically for the interface, the region where error is largest — beyond 60° — lies well past the 30° trigger, so gesture reliability is governed by the two innermost bands, where error is 4.77° and 5.09° against an 18° hysteresis band. Expressing this as a margin,

$$
margin = ( θ_on − θ_off ) / ( 2 · e_band ) = 18° / ( 2 × 5.09° ) = 1.77 ,
$$ (238)

so a typical error must be nearly doubled before it can carry a pose across the band.

Fig. 22 gives the per-face view.

!FIG headpose/H4_win_skyline.png | Fig. 22. Head-to-head skyline — every AFLW2000 face sorted by its per-face error gap.

Beyond the 14.58 % win rate the extremes carry a caution. Catastrophic failures above 30° number 63 for the proposed network (3.20 % of faces, contributing 1.36° of its 7.37° MAE) and 9 for the baseline (0.46 %, 0.23° of 3.48°). Their contribution to the aggregate is

$$
contribution = (1/n) Σ_{i ∈ 𝒞} e(i) ,   𝒞 = { i : e(i) > 30° } ,
$$ (239)

and excluding them leaves 6.01° against 3.25°, so the gap is not outlier-driven. But the baseline's single worst error (119.6°) *exceeds* the proposed network's (92.6°). The more accurate model owns the larger extreme, which argues against selecting architectures on maximum-error criteria over test sets of this size — the maximum of n draws is an inconsistent statistic whose expectation grows with n for any heavy-tailed error law — and which matters here because a single 90°+ error inside a dwell window is exactly the event the 1,200 ms cooldown exists to contain.

Fig. 23 stratifies by yaw band, where the shrinkage account is directly testable.

!FIG headpose/H5_yaw_ridgeline.png | Fig. 23. Error versus profile — error distributions per yaw band, from frontal through to full profile.

Yaw-axis MAE runs 6.11°, 6.98°, 9.78° and 18.43° across the four bands, while the error predicted by gain deficiency alone,

$$
ê_shrink(band) = ( 1 − ŝ_yaw ) · 𝔼[ |θ_yaw| | band ] = 0.178 · 𝔼[ |θ_yaw| | band ] ,
$$ (240)

runs 1.13°, 3.89°, 7.99° and 12.64°. The explained fraction

$$
ê_shrink / e_observed :  18 %  →  56 %  →  82 %  →  69 %
$$ (241)

rises from 18 % of observed error at frontal to a dominant share beyond 30°. Shrinkage is thus a minor contributor near frontal and the dominant one at profile, accounting for the accelerating degradation without any additional failure mechanism. The distributions broaden and shift together rather than becoming bimodal; bimodality would indicate sign confusion from flip augmentation without the label inversion of (26), and its absence confirms the augmentation logic is correct.

The deployment consequence is precise and follows from (130):

$$
θ_true^trigger = θ_on / ŝ_yaw = 30° / 0.822 = 36.5° ,
$$ (242)

so a user must turn to roughly 36.5° of true yaw to cross a 30° trigger. This is a systematic, stable and learnable offset — users adapt to it within a session — rather than an erratic one, which is why the gesture layer remains usable at an 8.74° yaw MAE.

Fig. 24 gives the regression fits.

!FIG headpose/H6_true_vs_pred.png | Fig. 24. Predicted versus true — density-weighted fits with coefficient of determination for yaw, pitch and roll.

Explained variance by (153) is 87.5 % for yaw, 61.8 % for roll and 58.2 % for pitch, against 98.0 %, 88.6 % and 80.2 %. Read alongside the fitted slopes (225) the two statistics separate cleanly: R² measures how much structure the network recovers, the slope whether it recovers it at the correct scale. Formally, by (212) the two are related through

$$
R² = ŝ² σ_θ² / ( ŝ² σ_θ² + σ_ζ² ) ,
$$ (243)

so a high R² with a low ŝ is the signature of *low residual variance at contracted scale* — the profile of an under-confident rather than an uninformed estimator. The proposed network scores comparatively well on the first and poorly on the second, and that is the reason a fixed trigger threshold rather than a learned one remains adequate: an under-confident estimator's output is a deterministic rescaling of the truth, which a fixed threshold accommodates by (242).

Fig. 25 locates the winner in pose space.

!FIG headpose/H7_win_map.png | Fig. 25. Pose battleground — every face placed at its true angle and coloured by which model achieves the lower error.

Winning regions interleave rather than separating territorially, exactly as constant-ratio degradation predicts: with no pose region of systematic advantage, the wins are per-sample noise inside a dominated comparison. A pose-conditioned switching policy — superficially attractive as a route to baseline accuracy at reduced average cost — has no support here, and the r_AR = 0.671 error correlation independently confirms it would have little to exploit, since by (164) the achievable gain is bounded by 7.5 % even with an oracle.

Figs. 26–29 re-render the same data as continuous fields, testing the joint claims of constant ratio, zero bias and benchmark sparsity.

!FIG figures_art/headpose/HA1_atlas.png | Fig. 26. Atlas of poses — every AFLW2000 face charted at its true head angle and lit according to which model achieves the lower error.

The continuous counterpart of Fig. 25 makes the absence of territory explicit: the 14.58 % of faces won scatter with no contiguous region of advantage. Under a 2.03–2.21× ratio across the manifold by (237), scattered wins are what per-sample noise in a dominated comparison produces, and the r_AR = 0.671 correlation confirms the two models fail on the same faces. Two independent measurements — one spatial, one rank-based — converge on the same conclusion about fusion.

!FIG figures_art/headpose/HA2_butterfly.png | Fig. 27. Butterfly of drift — per-axis signed-error distributions, proposed network as the upper wing and the pretrained baseline as the lower wing of each pair.

Every wing is pinned on zero by (229), so the deficit is width, not displacement. Spread ratios are

$$
σ^A/σ^R:  2.49×  (yaw) ,  1.46×  (pitch) ,  1.84×  (roll) .
$$ (244)

That the yaw ratio is largest while yaw is also the best-explained (R² = 0.875) and least contracted (ŝ = 0.822) axis is the visual statement of the units artefact of (234) — a modest relative deficit over a wide range produces the widest absolute distribution. The pinning on zero is what licenses the dwell-averaging argument of (230)–(231).

!FIG figures_art/headpose/HA3_silk.png | Fig. 28. Silk streams — error flowing as a continuous function of head rotation, from full profile through frontal and back.

Continuity is the finding: the stream swells smoothly from 4.77° at frontal to 18.16° beyond 60° with no discontinuity or bifurcation, so the progressively occluded operator identified in Fig. 31 below degrades gracefully rather than failing abruptly. A discontinuity would signal a regime change, a bifurcation sign confusion; neither appears. Graceful degradation is what permits the gesture layer to stay stable at large angles, because error grows predictably with a state the state machine already tracks.

!FIG figures_art/headpose/HA4_honeycomb.png | Fig. 29. Honeycomb of error — the yaw × pitch plane tiled into cells, each glowing according to the mean error of the faces it contains.

Tiling restores what smooth interpolation suppresses: cell occupancy. The brightest peripheral cells are the sparsest — 278 of 1,969 faces lie beyond 60° of yaw against 891 within 15° — so extreme-pose estimates rest on roughly a seventh of the evidence supporting frontal ones. The standard error of a band mean scales as

$$
SE(band) = σ_band / √n_band ,   SE(>60°)/SE(<15°) = ( σ_{>60}/σ_{<15} ) · √( 891/278 ) ≈ 1.79 · ( σ ratio ) ,
$$ (245)

so the region where the models differ most is the region the benchmark constrains least, and the 18.16° against 8.20° comparison there carries materially wider uncertainty than its frontal counterpart. This caveat applies to AFLW2000-based comparisons generally, and it is reassuring for this system only because the operative range lies inside the well-populated bands.

#### 5.3.1 What the network computes

!FIG attribution/headpose/A1_summary.png | Fig. 30. Head pose — global attribution summary for predicted yaw over the 64 × 64 face crop.

The attribution field is antisymmetric about the facial midline: columns 1–2 predominantly positive (R2C2, R1C2, R3C1, R2C1), columns 3–4 predominantly negative (R2C3, R1C3, R2C4, R3C4), with mass in the brow–eye–nose band of rows 1–3. Formally, writing the region attribution vector as a function of the mirror operator ℳ,

$$
Attr( ℳ R_k ; x ) ≈ − Attr( R_k ; x )   for k in the brow–eye–nose band ,
$$ (246)

which is precisely the antisymmetric operator (216) already identified for gaze. Yaw is physically encoded by the left–right asymmetry of that band under projection, and the network recovered exactly that computation — structurally the same class of operator the gaze network learned, which is why both exhibit the same shrinkage behaviour: a differential operator's output amplitude falls as one of its two arguments is occluded, and (155) then contracts the estimate. Row 4, jaw and chin, contributes little, a sensible allocation given it is the region most degraded by self-occlusion at profile.

!FIG attribution/headpose/A2_waterfall.png | Fig. 31. Head pose — waterfall attribution for two extreme instances of predicted yaw.

At extreme yaw the decomposition is dominated by one side of the antisymmetric pair, the occluded side contributing little. This is the mechanism generating the gain deficit of Fig. 23. Modelling the differential operator's effective gain as a function of the visible fraction v(θ_yaw) ∈ [0,1] of the occluded hemiface,

$$
s_eff( θ_yaw ) = σ_θ² / ( σ_θ² + σ_ε²/v(θ_yaw) ) ,   v decreasing in |θ_yaw| ,
$$ (247)

so that as yaw increases, half the differential measurement is lost to self-occlusion, effective gain falls, and the estimate compresses toward the frontal prior. Equation (247) explains why contraction *worsens with angle* rather than staying constant, matching the accelerating explained fraction of (241), and why no bimodality appears — the operator degrades continuously rather than switching sign.

!FIG attribution/headpose/A3_similarity.png | Fig. 32. Head pose — global explanation under similarity ordering.

Similarity ordering traces a smooth monotone progression of region weights following the pose manifold. No discontinuity appears anywhere along it, corroborating the constant-ratio degradation of Fig. 21 from an entirely independent measurement: there is no pose regime in which the network engages a qualitatively different strategy, only one in which the same strategy has less evidence available — which is exactly the content of (247).

!FIG attribution/headpose/A4_ordering.png | Fig. 33. Head pose — global explanation under dataset ordering over AFLW2000.

In dataset order the pattern is heterogeneous, reflecting AFLW2000's unordered pose composition. The absence of structure is the intended control: it confirms the coherent progression under similarity ordering in Fig. 32 is a property of the explanation manifold rather than an artefact of acquisition sequence, a distinction a single ordering could not have drawn.

### 5.4 Voice — command execution

Voice runs on its own audio thread, outside the camera loop's frame budget (3), and maps six keywords to the commands that make the rest of the interface operable: *up* and *down* scroll, *stop* and *go* pause and resume control, *yes* and *no* confirm and cancel. Its row in Table 6 names per-class precision on the safety-critical words as the governing statistic, and the reason is consequence rather than frequency — by (135), *go* is the only route back once *stop* has paused the system, so an error on it must be recovered by repetition against a system that is deliberately not acting. This section therefore reports per-class behaviour before aggregate behaviour, inverting the usual order.

On the 2,468-clip official test split the proposed network attains

$$
Acc = 94.53 % ,  BalAcc = 94.48 % ,  macro-F₁ = 94.48 % ,  MCC = 0.9344 ,  κ = 0.9344 ,
$$ (248)

against the production ASR baseline at 95.50 % accuracy and 97.43 % macro-F₁. The 2.95 pp macro-F₁ gap is produced by a *mechanism* rather than by discriminative superiority, and separating the two is the substance of this section.

Fig. 34 presents the error structure.

!FIG voice/V1_confusion.png | Fig. 34. Confusion matrix — row-normalised, showing where the clips of each spoken keyword were assigned.

The 135 errors concentrate in three pairs: *down* ↔ *go* (38 clips), *go* ↔ *no* (30) and *down* ↔ *no* (17), together

$$
( 38 + 30 + 17 ) / 135 = 63.0 %
$$ (249)

of all errors. All three involve a back rounded nucleus and differ principally in a brief initial consonant, so the discriminative evidence occupies a few tens of milliseconds at onset — a small number of the T = 81 time frames produced by (28). The precision–recall split localises the mechanism:

$$
Prec(go) = 0.877 ,  Rec(go) = 0.908 ;   Prec(down) = 0.948 ,  Rec(down) = 0.889 .
$$ (250)

Since Prec(go) < Rec(go) while the reverse holds for *down*, *go* is an **attractor**: it absorbs clips from its phonetic neighbours rather than being a class the network fails to recognise. Formally, writing N(c′→c) for the count of true-c′ clips assigned to c, the in-flow/out-flow imbalance

$$
Λ_c = Σ_{c′ ≠ c} N_{c′→c} − Σ_{c′ ≠ c} N_{c→c′}
$$ (251)

is positive for *go* and negative for *down*, which is the quantitative content of "attractor".

For the interface this is the most consequential single result in the voice evaluation, and it is a **design fault rather than a model fault**. *go* is bound to "resume control": the command that reactivates the interface after *stop* has suspended it. Its being an attractor means spurious resumes while the system is deliberately paused; its 0.877 precision is the worst in the vocabulary; and its partners in confusion are *down* (scroll) and *no* (escape), both frequently spoken. Meanwhile *stop*, the pause direction, is among the strongest words (precision 0.971, recall 0.976). The safety-critical asymmetry therefore runs the wrong way — the gate is easy to close and easy to open accidentally — and the remedy costs nothing: rebinding "resume control" to a phonetically distant token, one with fricative or stop-cluster onset (the acoustic class that Fig. 37 below shows the network separates best), removes the largest error mode without retraining a single parameter.

Fig. 35 places the proposed network against the ASR baseline across shared metrics.

!FIG voice/V2_metric_rose.png | Fig. 35. Metric rose — seven summary scores as paired petals, proposed network against the pretrained ASR baseline.

The baseline's macro precision is 99.45 % against 94.53 % while its macro recall is 95.53 % against 94.48 %, and the divergence is explained entirely by abstention. The engine rejects 98 of 2,468 clips, emitting no hypothesis, and among the 2,370 it does transcribe it commits only 13 substitutions:

$$
cov_R = 2370/2468 = 96.03 % ,   risk_R = 13/2370 = 0.55 % .
$$ (252)

It is not a more discriminative classifier but one equipped with a rejection option in the sense of (132), converting would-be substitutions into non-responses. Macro-F₁ rewards this — abstention costs recall but protects precision — whereas forced-choice accuracy, the like-for-like comparison under (131), differs by only

$$
95.50 % − 94.53 % = 0.97  pp .
$$ (253)

In this interface the distinction is operational rather than semantic: under the asymmetric loss (134) a substitution executes an unintended command while an abstention costs one repetition, and given the *go* binding above the abstention is clearly preferable.

Fig. 36 asks whether the same rejection regime is available to the proposed network, which requires its confidence to be a probability rather than a ranking.

!FIG voice/V3_constellation.png | Fig. 36. Calibration constellation — confidence bins as bubbles against observed accuracy on the honesty diagonal.

Expected calibration error by (150) over fifteen bins is

$$
ECE = 1.54 % ,   𝔼[ conf | correct ] = 0.950 ,   𝔼[ conf | error ] = 0.640 ,
$$ (254)

with macro one-versus-rest ROC-AUC 0.9961 and log-loss 0.1644. Selective prediction with a reject option is therefore available, and the risk–coverage pairs of (133) are

$$
ζ = 0.7:  cov = 92.2 % ,  Acc = 97.72 % ;   ζ = 0.9:  cov = 81.5 % ,  Acc = 99.10 % .
$$ (255)

The honest comparison is nonetheless unfavourable. Matching the baseline's 99.45 % precision requires

$$
ζ = 0.965 ,   cov = 68.9 %   against the baseline's  cov_R = 96.0 % ,
$$ (256)

so the proposed network *can* adopt the abstention strategy but attains a markedly worse risk–coverage curve. This, rather than the headline accuracy, is the substantive gap between the two systems. It also gives the deployment rule a defensible setting: a ζ = 0.7 gate costs 7.8 % coverage and removes roughly half the substitution risk, and by the Chow rule following (134) this corresponds to an implied cost ratio c_sub/c_abs ≈ 3.3, which is a defensible valuation for an interface in which an unintended command may require several actions to undo.

Fig. 37 decomposes confidence by keyword.

!FIG voice/V4_perword_ridgeline.png | Fig. 37. Per-word confidence — the distribution of predicted probability assigned to the correct class, per keyword.

One-versus-rest AUC orders as

$$
yes  0.9997 >  stop  0.9992 >  up  0.9984 >  no  0.9959 >  down  0.9941 >  go  0.9896 ,
$$ (257)

and the ordering is acoustic, not statistical: class frequencies span only 402–425 clips, so prior imbalance explains none of the spread,

$$
max_c n_c / min_c n_c = 425/402 = 1.057 .
$$ (258)

The two strongest carry unvoiced fricative or plosive energy — the /s/ of *yes*, the /st/ cluster of *stop* — producing a high-frequency signature unique among the six and sharply localised in time, hence concentrated in the upper mel bands of (31) and a narrow span of time frames. The weakest, *go*, is a single voiced plosive with a sustained vowel and no coda: the shortest window of class-discriminative evidence in the vocabulary. This is the quantitative basis for the rebinding recommendation of Fig. 34 — the network's weakness is inherited from the label set, so changing the label set is the correct intervention.

Fig. 38 quantifies separability independently of any threshold.

!FIG voice/V5_separation.png | Fig. 38. Separation densities — per keyword, the score distribution on target clips against all others, with AUC.

Every keyword exceeds 0.989 AUC, so all six classes are near-separable in the learned latent space at the ranking level, while operating-point accuracy is 94.53 %. The gap between the two,

$$
𝔼_c[ AUC_c ] = 0.9962   against   Acc = 0.9453 ,
$$ (259)

localises the residual error in **threshold placement and class-boundary geometry** rather than in feature learning. This is the diagnostic that separates a model needing capacity from one needing decision logic, and it places this network unambiguously in the second category. It is also why the confidence gate of (255) lifts accuracy to 99.10 % without touching a weight.

Fig. 39 examines the decision margin.

!FIG voice/V6_decision_margin.png | Fig. 39. Decision margin — the gap between the top two class probabilities, for correct against incorrect predictions.

Defining the margin

$$
m(x) = p_(1)(x) − p_(2)(x) ,
$$ (260)

the measured means are 0.912 on correct predictions and 0.404 on errors, so errors carry measurable internal hesitation and are detectable at runtime from logits already computed. A useful negative result attaches: as an error detector the margin achieves AUC 0.930 against 0.932 for top-1 confidence,

$$
| AUC_margin − AUC_conf | = 0.002 ,
$$ (261)

statistically indistinguishable at this sample size, so the theoretical advantage of margin over confidence — invariance to overall distribution sharpness — does not materialise here. The simpler top-1 gate of (132) is the correct implementation choice, and a margin-based gate would add complexity for nothing measurable.

Fig. 40 gives the aggregate flow.

!FIG voice/V7_word_flow.png | Fig. 40. Word flow — every clip streaming from its true keyword to its predicted keyword; crossings are confusions.

Misrouting concentrates in a few thick ribbons and the confusion graph is sparse. Treating the confusion counts as a weighted graph on the six keywords, the edge density is

$$
density = |{ (c, c′) : N_{c→c′} > 0, c ≠ c′ }| / ( C(C−1) ) ≪ 1 ,
$$ (262)

with most keyword pairs exchanging no clips at all. Sparsity matters more than total error mass for interface design, because a sparse graph is remediable by vocabulary substitution or by a targeted pairwise discriminator over three pairs, both far cheaper than retraining the six-way classifier. A dense graph of identical error mass would admit neither remedy — which is what makes the recommendation of Fig. 34 available at all.

Figs. 41–44 re-render the same predictions, testing the sparsity and confidence-separation claims as joint structure.

!FIG figures_art/voice/VA1_chord.png | Fig. 41. Confusion constellation — keywords as nodes sized by frequency, with arcs carrying the misrecognised clips between them.

Rendering confusion as connectivity shows that of fifteen possible undirected pairs, three carry 63 % of the 135 errors while several carry none, and that the three loaded arcs form a **connected triangle** among the back-vowel words {*down*, *go*, *no*}. That is a stronger statement than pairwise confusion: it identifies a single phonetic neighbourhood as the locus of all significant uncertainty. Removing one vertex from a triangle removes two of its three edges, which is why one substitution inside that triangle reduces total error disproportionately —

$$
Δ error ≥ ( 38 + 30 ) / 135 = 50.4 %  if go is rebound ,
$$ (263)

against the 28 % that removing an arbitrary single edge would achieve. The graph structure, not the error count, is what makes the remedy identifiable.

!FIG figures_art/voice/VA2_sunflower.png | Fig. 42. Confidence sunflower — every test clip as a seed, coloured by predicted keyword and sized by confidence, with misclassified clips ringed.

Ringed seeds are systematically small, the per-clip form of the 0.950 against 0.640 confidence separation of (254). The informative population is the exception: large ringed seeds are confidently wrong clips that no gate can intercept. Their scarcity relative to small ringed seeds is quantified by the 0.932 error-detection AUC of Fig. 39 and is what makes the operating points of (255) achievable — a model with identical accuracy but poorer confidence separation would admit no selective regime at all, and the abstention comparison with the ASR baseline could not even be posed.

!FIG figures_art/voice/VA3_starmap.png | Fig. 43. Word star-map — each clip's six-way probability vector projected to two dimensions; tight islands indicate confidently separated classes.

Projecting the output simplex Δ⁵ = { p ∈ ℝ⁶₊ : Σ p_c = 1 } to two dimensions visualises decision geometry directly. Six islands resolve, consistent with all per-word AUCs exceeding 0.989, and their tightness follows the AUC ordering (257) — *yes* and *stop* most compact, *go* most diffuse and most overlapped with its neighbours. Because the projection is of the *probability vector* rather than an internal embedding, island overlap corresponds directly to realisable confusion at the decision boundary, connecting latent geometry to the confusion structure of Fig. 34 without an intervening assumption.

!FIG figures_art/voice/VA4_aurora.png | Fig. 44. Aurora of the keywords — per keyword, the density of probability assigned to the correct class.

The curtains rank as the per-class F₁ scores do,

$$
yes 0.987 > stop 0.973 > up 0.960 > no 0.939 > down 0.917 > go 0.892 ,
$$ (264)

while class frequencies span only 402–425 clips by (258), so frequency explains none of the spread and phonetic distinctiveness explains all of it. This is the cleanest available statement that residual voice error is a property of the chosen command vocabulary — addressable by vocabulary design at zero computational cost rather than by additional capacity, which for a system budgeted at 101,766 parameters by (84) is the difference between a solvable and an unsolvable problem.

#### 5.4.1 What the network listens to

!FIG attribution/voice/A1_summary.png | Fig. 45. Voice — global attribution summary over the 40 × 81 log-mel spectrogram, regions indexed by mel band and time slot.

Attribution concentrates in the upper mel bands (mel3, mel4) during central time slots (t2, t3), with negligible contribution from the lowest bands and the leading and trailing slots. The network therefore learned an **implicit voice-activity gate**: Speech Commands clips are one second long by (27) with the utterance roughly centred and silence-padded, and the model learned to discard the padding without being told where the word begins. Writing the region attribution as a separable product,

$$
Attr( b, τ ) ≈ ω(b) · g(τ) ,   g(τ) ≈ 0  for τ in the leading and trailing slots ,
$$ (265)

the temporal factor g is the gate and the spectral factor ω the evidence weighting. Both matter for deployment: g is why the network transfers to a live microphone stream whose padding statistics differ from the corpus, and ω — concentrated in the high-frequency bands carrying fricative and plosive energy — is the mechanistic counterpart of the AUC ordering (257) and the reason *go*, whose discriminative energy is a brief low-frequency transient, is weakest.

!FIG attribution/voice/A2_waterfall.png | Fig. 46. Voice — waterfall attribution for two extreme instances of the predicted keyword.

Per-instance decompositions are dominated by two or three time-localised high-band regions, so classification rests on short sparse spectro-temporal events rather than integration over the clip. Writing the effective support as

$$
| supp( Attr ) | / |ℛ| ≈ 3/16 = 0.19 ,
$$ (266)

this sparsity is what permits 101,766 parameters to reach 94.53 % at 2.7 ms per clip — an efficiency result with an explanation rather than an empirical accident, since by (78) global average pooling discards spatial extent precisely because the evidence is sparse and its location is not informative. It also predicts the failure mode: where the discriminative event is short and low-energy, as for *go*, the margin (260) collapses, which is exactly what Fig. 39 measures and Fig. 34 localises.

!FIG attribution/voice/A3_similarity.png | Fig. 47. Voice — global explanation under similarity ordering.

Similarity ordering produces clearly separated blocks, in contrast to the continua of gaze and head pose. This is the expected structure for a categorical task — the explanation map (217) is Lipschitz with respect to a *discrete* label metric rather than a continuous target — and it is a cross-check that the ordering reflects task structure rather than an artefact of the method. Each block corresponds to clips explained by the same regions, and clips at block boundaries are candidates for the confusable pairs of Fig. 34, linking explanation structure directly to confusion structure.

!FIG attribution/voice/A4_ordering.png | Fig. 48. Voice — global explanation under dataset ordering over the official test list.

Explanations are stationary across the speaker-disjoint official test list, which is ordered independently of class and speaker, indicating speaker-invariant evidence regions: the network is not adapting its strategy to particular voices. Formally, regressing region attribution on test-list position yields no significant trend,

$$
| corr( Attr(R_k ; xᵢ) , i ) | < ϵ  for all k ,
$$ (267)

which demonstrates mechanistically what the official split guarantees only by construction, and confirms that the 84.72 % → 94.53 % gain of Table 11 reflects acoustic modelling rather than speaker-specific fitting — the property a single-user assistive deployment depends on entirely.
