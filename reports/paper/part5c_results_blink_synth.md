### 5.5 Blink — click generation

Blink is the click, and it is the only modality with no pretrained neural comparison: as recorded in Section 4.2, the practical alternative is a geometric eye-aspect-ratio rule requiring full-face landmarks, which cannot be scored on 32 × 32 cropped eye regions, and no estimated figure is substituted in its place. Blink is therefore reported as a per-sample audit over 12,927 held-out eyes (4,453 open, 8,474 closed), which is the evidence required to set its operating threshold.

Table 6 makes the framing unusually important here. A click is not one classification; it is a closure sustained past 420 ms — n_run = 13 consecutive frames at 30 fps by (120) — with natural blinks of 100–300 ms rejected by construction. The governing quantity is therefore the integrity of a multi-frame run, not per-frame accuracy, and this section ends by showing that this **inverts** the cost asymmetry the raw confusion matrix suggests. Headline metrics, by (140)–(150):

$$
Acc = 93.67 % ,  BalAcc = 93.32 % ,  macro-F₁ = 93.04 %  ( 0.909 open, 0.951 closed ) ,
$$ (268)

$$
MCC = 0.8610 ,  κ = 0.8608 ,  ROC-AUC = 0.9836 ,  PR-AUC = 0.9904 ,  Brier = 0.0469 ,  LogLoss = 0.1700 ,
$$ (269)

with MCC as defined in (144) [@matthews] and κ in (145) [@cohenkappa].

Fig. 49 shows the raw score distribution.

!FIG blink/B1_swarm.png | Fig. 49. Separation swarm — every test eye positioned by its predicted probability of the closed state.

Two clouds occupy opposite extremes with a sparse interior, the visual form of 0.9836 ROC-AUC. The gap between ranking quality and operating-point accuracy is the actionable quantity:

$$
ROC-AUC − Acc = 0.9836 − 0.9367 = 0.0469 ,
$$ (270)

and a classifier ordering this well but scoring materially lower is *misthresholded*, not underpowered. By the probabilistic identity (146) the only samples that can change class under any threshold move are those in the overlap region, so the sparse interior also bounds what threshold tuning can recover — which is why the threshold sweep of Fig. 52, rather than a retraining effort, is the right next step.

Fig. 50 accounts for class prevalence.

!FIG blink/B2_mosaic.png | Fig. 50. Confusion mosaic — territory proportional to class prevalence; column width is prior, solid area is correct classification.

At 65.5 % closed, accuracy is prevalence-weighted toward the majority class, so balanced accuracy (141) and MCC (144) are the appropriate summaries and both remain high; performance is therefore not a prior artefact. The mosaic exposes the asymmetry governing deployment:

$$
FP = 347  open eyes classified closed  ( 7.79 %  of open ) ,   FN = 471  closed eyes classified open  ( 5.56 %  of closed ) .
$$ (271)

At the frame level these are spurious-click and missed-blink pressure respectively; what the state machine of Algorithm 5 makes of them is the subject of Fig. 52.

Fig. 51 tests probabilistic honesty.

!FIG blink/B3_constellation.png | Fig. 51. Calibration constellation — confidence bins as bubbles against observed accuracy.

Expected calibration error by (150) is

$$
ECE = 2.48 % ,   𝔼[ conf | correct ] = 0.973 ,   𝔼[ conf | error ] = 0.785 ,
$$ (272)

consistent with the 0.0469 Brier score. The residual miscalibration exceeds the voice network's 1.54 %, which is expected rather than anomalous: at 136,066 parameters on a binary task with a strong prior, the network sits in the regime where cross-entropy training drives logits toward saturation. Concretely, minimising (88) on separable data drives ‖f‖ → ∞ along the max-margin direction, and the induced posterior (87) approaches a step function; the deviation from calibration is therefore a predictable consequence of the objective rather than a defect of the fit. Calibration nonetheless remains adequate for the state machine's staged progress readout (Algorithm 5, line 11), which reports fractional progress toward each click stage and therefore requires per-frame scores to be commensurable *across* frames rather than merely correctly ordered *within* one.

Fig. 52 sweeps the decision threshold — the single most consequential deployment parameter for this modality.

!FIG blink/B4_streamgraph.png | Fig. 52. Outcome streamgraph — the four confusion outcomes trading off continuously as the decision threshold slides from 0 to 1.

The accuracy-optimal threshold is

$$
ζ_acc = arg max_ζ Acc(ζ) = 0.212 ,   Acc(0.212) = 94.35 %   against   Acc(0.50) = 93.67 % ,
$$ (273)

and the Youden-optimal threshold [@youden], weighting both error types equally in rate terms, is

$$
ζ_J = arg max_ζ [ TPR(ζ) − FPR(ζ) ] = 0.426 .
$$ (274)

Minimising the cost-weighted objective

$$
C(ζ ; w) = w · FP(ζ) + FN(ζ)
$$ (275)

for w ∈ {3, 5, 10} gives ζ* = 0.752, 0.845 and 0.955, cutting false positives from 347 to 226, 184 and 114 at the cost of 769, 955 and 1,492 missed blinks. The exchange is strongly non-linear:

$$
Δ FN / Δ FP :  298/121 = 2.46   then   186/42 = 4.43 ,
$$ (276)

so each successive false positive removed costs progressively more missed blinks — the local slope of the receiver-operating curve steepening as the operating point moves into the tail.

**The cost-weighted reading is the natural one, and for this system it is wrong.** The reason is (120)–(123). Because a click requires n_run = 13 consecutive frames above threshold, an isolated false positive cannot fire one; by (121) the probability that noise alone manufactures a click from an open eye is

$$
FPR^{n_run} = 0.0779^{13} ≈ 3.9 × 10⁻¹⁵ ,
$$ (277)

and the state machine's duration requirement is therefore itself a powerful false-positive filter, with the 100–300 ms natural-blink rejection removing the most common sustained closure as well. A false *negative*, by contrast, breaks a run in progress and silently cancels a click the user deliberately made; by (122) the survival probability of a genuine run is

$$
( 1 − FNR )^{n_run} = 0.9444^{13} ≈ 0.475 ,
$$ (278)

so under independence more than half of deliberate clicks would be lost. Under the deployed decision rule the asymmetry therefore **inverts** relative to the frame-level intuition, and thresholds *below* 0.50 are the defensible choice: at ζ = 0.212 false negatives fall from 471 to 280, raising run survival to

$$
( 1 − 280/8474 )^{13} = 0.9670^{13} ≈ 0.646 ,
$$ (279)

a 36 % relative improvement in click retention, while spurious firing rises only to a still-negligible value by (277). The threshold ζ = 0.212 is thus simultaneously accuracy-optimal by (273) and run-integrity-optimal by (279).

This conclusion depends on consecutive-frame errors being less than perfectly correlated. Under the Markov model (123) with persistence ϱ the effective run length is n_eff^run = n_run(1−ϱ)/(1+ϱ), and both (277) and (279) must be recomputed at n_eff^run; the *direction* of the asymmetry is preserved for any ϱ < 1, but its magnitude is not. That assumption is not tested here, and a live-session measurement of click precision and recall would be required to confirm it, which Section 6 records as an open item.

Fig. 53 shows the same structure as a flow.

!FIG blink/B5_confusion_flow.png | Fig. 53. Confusion as a river — samples flowing from true to predicted state; cross-ribbons are errors.

The two cross-ribbons are comparably thick (347 and 471) despite a 2:1 class imbalance, so per-class error *rates* are asymmetric opposite to the prior:

$$
FPR = 347/4453 = 7.79 %   against   FNR = 471/8474 = 5.56 % ,
$$ (280)

with per-class F₁ of 0.909 (open) against 0.951 (closed). This inverts the usual imbalance expectation — the minority class is usually harder because it is scarcer — and identifies **intra-class variance** rather than prior as the governing factor: the open class spans a wide appearance range (gaze direction, illumination, reflective eyewear) while the closed class is comparatively canonical. Formally, if the two classes have equal Bayes-optimal separability but unequal conditional entropy, then

$$
H( X | Y = open ) > H( X | Y = closed )  ⟹  error rate_open > error rate_closed
$$ (281)

at any fixed capacity. Rebalancing the training prior would therefore not help — it addresses the wrong term — whereas enlarging the open-class appearance range would, and eyewear is the specific variation the deployment gap of Section 6 names.

Fig. 54 measures score quality independently of any threshold.

!FIG blink/B6_score_separation.png | Fig. 54. Score separation — how cleanly the predicted probability of the closed state divides the two classes.

PR-AUC (0.9904) exceeding ROC-AUC (0.9836) is the expected relation on a majority-positive problem [@davis], since the precision axis is inflated by a high positive prior, and it confirms the ranking is robust under the operative class prior rather than only under a balanced one — a distinction that matters because the deployed prior, dominated by open eyes during normal use, differs markedly from the test prior of 65.5 % closed. Separation of this quality places the residual 6.33 % error in genuinely ambiguous inputs: mid-blink frames where the eyelid is physically part-closed and the binary label is a quantisation of a continuous state. Writing the true eyelid aperture as a latent continuous variable a ∈ [0,1] with the label y = 𝟙[a < a₀], the conditional Bayes risk near the boundary is

$$
ℙ[ error | a ≈ a₀ ] → ½ ,
$$ (282)

irreducibly, so that component of the error cannot be removed at the per-frame level by any classifier. This is precisely the argument for resolving it temporally, which is what the duration requirement of (118) does.

Fig. 55 tests whether the network signals its own errors.

!FIG blink/B7_margin_ridge.png | Fig. 55. Decision certainty — the certainty distribution of correct against incorrect predictions.

Confidence is a strong but incomplete error detector, AUC 0.9148, and the quintile profile makes the limitation precise. Error rates across confidence quintiles run

$$
26.60 % ,  3.95 % ,  0.85 % ,  0.19 % ,  0.04 % ,
$$ (283)

a spread of

$$
26.60 / 0.04 = 665×
$$ (284)

from lowest to highest. Yet

$$
ℙ[ conf > 0.9 | error ] = 31.2 % ,
$$ (285)

so roughly one error in three is *confidently wrong* and unreachable by any per-frame gate. This is the quantitative justification for the state machine's duration requirement rather than a confidence threshold: mid-blink ambiguity in the sense of (282) is transient while a genuine closure persists, so consecutive-frame agreement suppresses exactly the confident isolated errors that per-frame confidence cannot. Formally, a per-frame gate at level ζ_c reduces error by at most 1 − ℙ[conf > ζ_c | error], bounded below by 0.312 here, whereas the run requirement suppresses isolated errors at rate governed by (277) irrespective of their confidence.

Figs. 56–59 re-render the same scores, testing the locality of misordering and the shape of the threshold trade.

!FIG figures_art/blink/BA1_delta_stream.png | Fig. 56. The delta of decision — two facing rivers of predicted score, closed eyes flowing upward and open eyes downward, parting at the decision threshold.

The rivers are near-disjoint except at a narrow waist, the geometric expression of 0.9836 ROC-AUC and 0.9904 PR-AUC, and all 6.33 % of the error lives in that waist — which is why threshold placement rather than representation is the operative parameter. The asymmetry about the parting line previews the cost analysis: the closed-eye river is the larger (8,474 against 4,453), so shifting the threshold to suppress false clicks sacrifices a disproportionate number of closed-eye detections, which is the non-linear exchange rate measured in (276).

!FIG figures_art/blink/BA2_galaxy.png | Fig. 57. Blink galaxy — all 12,927 test eyes arranged on a spiral in order of predicted score and coloured by their true state.

Ordering by score and colouring by truth renders ranking quality *countable*. By the identity (146), ROC-AUC is exactly the probability that a randomly chosen closed eye outranks a randomly chosen open one, so contaminated arc length relative to the whole spiral is a direct visual estimate of

$$
1 − AUC = 0.0164 .
$$ (286)

Contamination is confined to a short arc at the crossover rather than distributed along the spiral, establishing that misordering is **local** — the precondition for threshold tuning to be effective at all, and therefore the justification for treating Fig. 52 as the primary improvement lever. Had misordering been distributed, no threshold would recover the gap (270) and retraining would be the only route.

!FIG figures_art/blink/BA3_threshold_aurora.png | Fig. 58. Threshold aurora — the four confusion outcomes as continuous bands, as the decision threshold glides from 0 to 1.

Continuous rendering exposes gradients the sampled points of Fig. 52 cannot. Accuracy peaks at 94.35 % near ζ = 0.212 and declines to 93.67 % at 0.50 while the false-positive band contracts monotonically throughout, so

$$
dAcc/dζ < 0  and  dFP/dζ < 0  simultaneously on  ζ ∈ [0.212, 1] ,
$$ (287)

meaning the two objectives move in opposite directions with no crossing point and **no threshold is jointly optimal**. The cost-weighted optima of (275) are then read directly off the band gradients. The figure converts an implicit design assumption — how much worse is a false click than a missed blink — into an explicit tunable with a measured response curve, which is what allows the conclusion of (277)–(279) to be revised once the state machine is taken into account rather than argued in the abstract.

!FIG figures_art/blink/BA4_certainty_arc.png | Fig. 59. Arc of certainty — every test eye positioned along an arc by its predicted score, with errors highlighted.

Errors concentrate near the arc's centre where certainty is lowest, the basis for uncertainty-aware gating, with error rates falling across quintiles by (283). The counter-population limits the approach: by (285), 31.2 % of errors sit at the extremes where no gate reaches them. Per-frame uncertainty filtering therefore has a hard floor, and the deployed state machine aggregates across consecutive frames precisely because temporal persistence discriminates where per-frame certainty cannot.

#### 5.5.1 Why this modality generalises across subjects

!FIG attribution/blink/A1_summary.png | Fig. 60. Blink — global attribution summary: regions of the 32 × 32 eye crop ranked by mean absolute attribution, with per-sample distributions and the three strongest regions outlined on the mean input.

Attribution concentrates in R3C2, R3C3 and R2C2 — the central and lower-central band containing the palpebral fissure — while the four corner regions contribute near-zero attribution with tightly clustered per-sample distributions:

$$
Σ_{k ∈ centre} | Attr(R_k) | / Σ_k | Attr(R_k) |
≫
Σ_{k ∈ corners} | Attr(R_k) | / Σ_k | Attr(R_k) | .
$$ (288)

The network located the anatomically correct evidence region without spatial supervision and, more importantly, learned to *ignore* the periphery where hair, spectacle frames and background offer identity-correlated shortcuts. This is the mechanistic basis of its cross-subject robustness: a detector attending only to the fissure is structurally invariant to the appearance factors the leaked-partition evaluation of Section 5.1.1 was silently exploiting. Formally, if the attribution mass on identity-correlated regions is negligible, then the decision function is approximately measurable with respect to the fissure sub-σ-algebra, and identity enters only through it.

!FIG attribution/blink/A2_waterfall.png | Fig. 61. Blink — waterfall attribution for the two most extreme instances, carrying the network from baseline output to prediction region by region.

Completeness (138) makes this an audit rather than an illustration: the bars sum to the full logit displacement F(x) − F(0). For both extremes two or three central regions dominate, the remaining thirteen contributing a small alternating-sign residual, so the decision is effectively low-dimensional in region space:

$$
effective dimension ≈ | { k : |Attr(R_k)| > 0.1 · max_j |Attr(R_j)| } | ≈ 3  of  16 .
$$ (289)

This is why 136,066 parameters suffice and why inference costs 0.42 ms — the capacity requirement is set by the intrinsic dimensionality of the evidence, not by the input dimension, and that is what leaves frame budget for the other three modalities under (165).

!FIG attribution/blink/A3_similarity.png | Fig. 62. Blink — global explanation under similarity ordering: stacked region attributions with instances ordered by resemblance of their explanations.

Similarity ordering yields contiguous **blocks** rather than a gradient, indicating a small number of discrete strategies rather than one continuously varying rule — the categorical-task structure also seen for voice in Fig. 47, and the opposite of the continua of gaze and head pose. The blocks align with the two classes and, within them, with the *sign pattern* of the central regions, which is evidence that eye state is encoded through the polarity of a localised contrast measurement rather than a distributed template:

$$
F_blink(x) ≈ w · sgn( ⟨ x, ϕ_upper ⟩ − ⟨ x, ϕ_lower ⟩ ) ,
$$ (290)

approximately. A polarity code is cheap to compute and inherently robust to global illumination scaling, since a multiplicative photometric change leaves the sign of a difference unaltered — consistent with the normalisation (15), with stability across the illumination range present in the corpus, and with the modality's tolerance of the variable lighting a home deployment presents.

!FIG attribution/blink/A4_ordering.png | Fig. 63. Blink — global explanation under dataset ordering: the same stacked attributions with instances in acquisition order.

In acquisition order the pattern is stationary, with no drift in region importance across the test sequence. Since the corpus is ordered by subject, stationarity under this ordering is a **direct test of subject-invariance** of the decision mechanism in the sense of (267), and it passes: the same evidence supports every held-out subject. This is the attribution-level counterpart of group-disjoint evaluation, and it converts 93.67 % from a number that survived a leakage check into one with an identified invariant mechanism behind it — the property on which deployment to an unseen user depends.

### 5.6 What the four analyses jointly establish

The four modality sections were deliberately reported in the same order — headline comparison, mechanism, diagnostic figures, synthesis figures, attribution — so that their conclusions could be compared rather than merely accumulated. Four findings hold across them.

#### 5.6.1 One mechanism governs both regression modalities

The fitted shrinkage gains of (191) and (225) are

$$
gaze:  ŝ = ( 0.483, 0.481 ) ;   head pose:  ŝ = ( 0.822, 0.455, 0.490 ) ,
$$ (291)

placing both networks in the shrinkage regime of (155) on every narrow-support axis. That single parameter accounts for:

- the isotropic radial growth of gaze error (Figs. 4, 13), via the systematic term (1 − s)‖θ‖ of (194), which has no azimuthal dependence;
- the constant 2.03–2.21× head-pose ratio across the pose manifold (Fig. 21), via the multiplicative form (211);
- the axis-ranking inversions between absolute and variance-normalised metrics (Figs. 10, 20), via the competing factors of (234);
- the tolerance convergence of Fig. 6, via the concentration of contraction cost in the tight-tolerance regime;
- the absence of any localised failure mode in either modality, since (194) is a smooth function of ‖θ‖ alone.

Yaw at ŝ = 0.822 is the exception that confirms the account: it is the only wide-support axis, and by the monotonicity (227) it is the only one reproduced at near-correct scale. That one measurement replaces what would otherwise be a dozen independent observations is the strongest evidence that the analysis has found the *mechanism* rather than described the *symptoms*.

#### 5.6.2 The contraction is rational, which redirects the engineering effort

Inverting the fitted gain raises gaze angular error by 28.5 % by (198), which by (157) establishes that both networks sit near their minimum-MSE shrinkage point s*. Two engineering consequences follow immediately.

First, **no post-hoc calibration improves them**; only a better representation does, since by (156) the risk at the optimum is MSE(s*) = s*σ_ε², a quantity that depends only on the representational noise σ_ε² and not on any downstream affine transform. This is why Section 5.2.4's identity-count ceiling is decisive rather than merely interesting: it closes the representation route on the available corpus, by (222).

Second, **the deployed pipeline's structure is correct**. Per-user calibration (106) restores scale — necessarily amplifying noise by the factor 1/ŝ ≈ 2.07 — and the temporal filtering (112)–(117) absorbs the noise that restoration releases. The apparent tension between a calibration stage that amplifies noise and a filter chain that suppresses it is not a design inconsistency but the correct decomposition of the problem, and the analysis identifies it as such.

#### 5.6.3 The classification modalities are limited by decision logic, not by features

The complementary diagnosis holds for blink and voice:

$$
ROC-AUC:  0.9836  (blink) ,  0.9961  (voice)   against   Acc:  93.67 % ,  94.53 % .
$$ (292)

The gaps of 4.69 pp and 5.08 pp place the residual error in **threshold placement rather than separability**, and both sections demonstrate the consequence directly: a confidence gate lifts voice accuracy to 99.10 % without changing a weight by (255), and a threshold move lifts blink accuracy to 94.35 % by (273). The attribution views explain why the features are adequate — all four networks compute task-appropriate operators:

- a signed differential across the midline for gaze and head pose, (216) and (246);
- a central-fissure contrast polarity for blink, (290);
- a gated high-band spectro-temporal detector for voice, (265).

so accuracy rests on identified transferable mechanisms rather than dataset-specific shortcuts. The two categories of limit call for different work, and distinguishing them is the practical payoff of the analysis: capacity for a representation-limited model, decision logic for a threshold-limited one, and the diagnostic (292) tells which is which before any effort is spent.

#### 5.6.4 Fusion with the pretrained baselines is not worthwhile

A fourth conclusion is negative and applies to both comparable modalities. Per-sample error correlations of

$$
r_AR = 0.491  (gaze) ,   0.671  (head pose)
$$ (293)

with oracle-selection bounds (163) of 3.303° and 3.222° give achievable fusion gains of only

$$
Gain_max = 8.8 %  (gaze) ,   7.5 %  (head pose) ,
$$ (294)

which are *upper* bounds requiring an oracle that knows per sample which model is right. The models fail on the same samples, so no gating or ensembling rule can recover the gap, and the cost of a second network — which by (165) would consume a further 2.13 ms or 4.13 ms of a 33.3 ms budget already carrying three networks and a landmark inference — is not repaid. This removes the most obvious escape route from the recommendation of Section 5.7 and is the reason that recommendation is stated without hedging.

### 5.7 Accuracy–efficiency operating point

Everything above characterises the networks individually. The system-level question is different: whether accuracy of this quality can be obtained inside a budget that lets four modalities run at once on a machine the user already owns. Table 12 reports the trade-off; all latencies are single-thread CPU measurements on the deployed TorchScript modules.

**Table 12.** Model complexity, footprint and single-thread CPU latency. Ratios are proposed relative to baseline.

| Modality | Model | Parameters | Size (MB) | Latency (ms) | Accuracy relative to baseline |
|---|---|---|---|---|---|
| Head pose | Proposed | 1,142,339 | 4.60 | 4.13 | 2.12× higher error |
| Head pose | 6-D rotation baseline | 39,323,398 | 157.32 | — | — |
| | *ratio* | ***34.4× fewer*** | ***34.2× smaller*** | | |
| Voice | Proposed | 101,766 | 0.44 | 2.7 | 0.97 pp lower accuracy |
| Voice | Production ASR engine | — | 70.90 | 26.5 | — |
| | *ratio* | | ***161× smaller*** | ***9.8× faster*** | |
| Blink | Proposed | 136,066 | 0.57 | 0.42 | no neural baseline exists |
| Gaze | Proposed | 551,810 | 2.23 | 2.13 | 1.78× higher error |
| Gaze | MPIIGaze pre-activation ResNet | 77,533 | 0.31 | — | — |
| | *ratio* | ***7.1× more*** | ***7.2× larger*** | | ***dominated*** |

Two qualitatively different outcomes are present and are reported separately rather than averaged into a single verdict.

#### 5.7.1 Defensible frontier positions: head pose and voice

For head pose and voice the proposed networks occupy defensible frontier positions. The relevant criterion is whether the accuracy penalty is *sub-linear* in the cost reduction by a wide margin — the condition for a compact model to be Pareto-relevant rather than merely small. Defining the efficiency elasticity

$$
η = log( error ratio ) / log( parameter ratio ) ,
$$ (295)

head pose gives

$$
η_pose = log(2.12) / log(34.4) = 0.751 / 3.538 = 0.212 ,
$$ (296)

so error grows as roughly the fifth root of the parameter reduction. Section 5.3 further showed that the residual error is *absorbed* by the dwell and hysteresis of the gesture rule — by (230)–(231) the dwell mean's spread is 2.8°–4.8° against an 18° hysteresis band — rather than reaching the user, so the elasticity understates the deployed advantage.

Voice operates within 0.97 pp of a production ASR engine at 1/161 of its storage and 9.8× its throughput, on a separate audio thread that does not consume the camera loop's budget at all, though Section 5.4 also showed its risk–coverage curve is markedly worse at matched precision by (256) and this is recorded as the substantive gap.

Aggregate vision-pipeline cost is, by (165),

$$
τ_nets = 0.42 + 2.13 + 4.13 = 6.68  ms ,   utilisation = 6.68 / 33.3 = 20.1 % ,
$$ (297)

leaving substantial headroom inside the 33.3 ms budget at 30 fps — headroom that matters because the same loop also runs the shared face-landmark inference, the gesture and blink state machines, and the cursor stabilisation chain. Total deployed footprint across all four modules is, by (166),

$$
S_total = 7.84  MB ,
$$ (298)

small enough to stay resident in last-level cache on typical consumer processors, which by the arithmetic-intensity argument of (167) is disproportionately important on CPU where inference at these sizes is bound by memory traffic rather than arithmetic.

#### 5.7.2 An unqualified negative result: gaze

For gaze no such argument survives, and this is recorded as an unqualified negative result. The pretrained baseline is simultaneously

$$
more accurate  ( 3.621°  against  6.439° )   and   smaller  ( 77,533  against  551,810  parameters,  7.1×  fewer ) ,
$$ (299)

dominating on both axes at once. In the Pareto sense there is no operating point at which the proposed gaze network is preferable: for any weighting w ∈ [0,1] of normalised error against normalised cost,

$$
w · ẽ_A + (1−w) · c̃_A > w · ẽ_R + (1−w) · c̃_R   for all  w ∈ [0,1] ,
$$ (300)

since both terms are strictly larger. Section 5.6.4 showed fusion would recover at most 8.8 %; Section 5.2.4 established the deficit is not closable by augmentation or additional sampling from this corpus. **The correct engineering decision is to adopt the pretrained model for gaze.**

#### 5.7.3 The protocol fault changes the decision

It is worth stating plainly that this conclusion was reachable only because the exposure-disjoint matched per-sample protocol of Section 3.5 made the comparison trustworthy. Under the original protocol the proposed gaze network scored

$$
2.84°   against the pretrained baseline's all-participant figure of   2.835° ,
$$ (301)

apparently level, and apparently vindicating the decision to train from scratch. Under the corrected protocol the true held-out contrast is

$$
6.439°   against   3.621° ,
$$ (302)

and the baseline is simultaneously 7.1× smaller. The decision inverts. The magnitude of the illusion is the quantity Theorem 3 decomposes: of the 1.905° inflation by (171), between 0.425° and 1.068° — 22 % to 56 % — is familiarity rather than difficulty by (174)–(175).

A protocol fault that changes an architectural decision is not a bookkeeping matter. We know of no sharper available demonstration that the distinction between a per-pipeline audit and a per-comparison audit has engineering consequences, and it is the reason the audit of Section 3.5 is presented as the principal contribution of this work rather than as a methodological aside.
