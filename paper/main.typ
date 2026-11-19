#set page(
  paper: "us-letter",
  margin: (x: 0.85in, y: 0.85in),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8.5pt, fill: rgb("#4b5563"))
      Generalized Higher-Dimensional EWMA Bias Correction: Operator Theory & Adaptive Optimization
      #h(1fr)
      Andrew Rober (2026)
      #line(length: 100%, stroke: 0.4pt + rgb("#d1d5db"))
    ]
  }
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.1")

#let thmbox(title, body) = block(
  width: 100%,
  fill: rgb("#f8fafc"),
  stroke: (left: 2.5pt + rgb("#1d4ed8"), rest: 0.5pt + rgb("#cbd5e1")),
  inset: (x: 10pt, y: 8pt),
  radius: 2pt,
  breakable: true,
)[
  *#title.* #emph(body)
]

#let proofbox(body) = block(
  width: 100%,
  inset: (top: 2pt, bottom: 4pt),
)[
  _Proof._ #body #h(1fr) $square.filled$
]

// Title Block
#align(center)[
  #set par(justify: false)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(6pt)
  #text(size: 16pt, weight: "bold")[
    Generalized Higher-Dimensional EWMA Bias Correction:\
    Non-Stationary Drift Elimination, the Blessing of Dimensionality,\
    and Adaptive Deep Learning Optimization
  ]
  #v(6pt)
  #line(length: 100%, stroke: 0.8pt + black)
  #v(8pt)
  #text(size: 11pt, weight: "bold")[Andrew Rober]\
  #text(size: 9.5pt, fill: rgb("#374151"))[
    Independent AI & Mathematical Optimization Research\
    `andrew_rober@hotmail.com` #sym.bullet `https://github.com/AndrewRober/HigherDimensionalEWMABiasCorrection`\
    November 2026
  ]
  #v(10pt)
]

// Abstract
#block(
  width: 100%,
  fill: rgb("#f3f4f6"),
  stroke: 0.5pt + rgb("#d1d5db"),
  inset: 11pt,
  radius: 3pt,
)[
  #align(center)[*Abstract*]
  #v(3pt)
  #set text(size: 9.3pt)
  The Exponentially Weighted Moving Average (EWMA) and its zero-initialization bias correction $hat(v)_t = v_t / (1 - beta^t)$ underpin modern signal processing and stochastic gradient optimizers (Momentum SGD, RMSprop, Adam, AdamW). Yet classical bias correction rests on three fragile assumptions: *(i)* zero initialization ($v_0 = 0$), *(ii)* stationary expectations ($EE[x_k] equiv mu$), and *(iii)* dense synchronous updates across all coordinates. Building upon and substantially generalizing our prior one-dimensional investigation @rober2023ewma, we formulate EWMA over arbitrary contraction operators $bold(B) in RR^(d times d)$ ($rho(bold(B)) < 1$) in $RR^d$ and prove five theoretical and algorithmic results:
  *(1) Affine-Invariant Operator Bias Correction & Duality (Theorem 1):* We derive the exact arbitrary-initialization operator estimator $hat(bold(v))_t^((0)) = (bold(I)_d - bold(B)^t)^(-1)(bold(v)_t - bold(B)^t bold(v)_0)$, its closed-form finite-$t$ covariance, and prove that under any anisotropic generator $bold(B)(epsilon) = exp(-epsilon bold(H))$, $hat(bold(v))_t^((0))$ interpolates between the instantaneous signal $bold(x)_t$ ($bold(B) -> bold(0)$) and the isotropic cumulative mean $macron(bold(x))_t = frac(1, t) sum_(k=1)^t bold(x)_k$ ($epsilon -> 0^+$).
  *(2) Exact Non-Stationary Lag & Directional Distortion (Theorem 2):* Under drift $bold(mu)_k = bold(mu)_t - (t-k)bold(g)$, standard bias correction suffers a permanent asymptotic lag $-bold(B)(bold(I)_d - bold(B))^(-1)bold(g)$ that rotates vector directions off-manifold in $RR^d$ when $bold(B)$ is anisotropic. Using a cascaded state $bold(u)_t$, we derive an $O(1)$-time closed-form finite-$t$ Trend-Corrected Operator EWMA ($hat(bold(v))_t^(("TC", 1))$) achieving zero bias for all $t >= 2$ and arbitrary $(bold(v)_0, bold(u)_0)$.
  *(3) The Blessing of Dimensionality—`AdaTC-EWMA` (Theorem 3):* Coupling $hat(bold(v))_t^((0))$ and $hat(bold(v))_t^(("TC", 1))$ via an online trend-invariant second-difference noise estimator $hat(tau)_t = frac(1, 6) norm(Delta^2 bold(x)_t)_2^2$ yields a hyperparameter-free James–Stein / Wiener shrinkage gate $hat(gamma)_t in [0, 1]$ whose excess per-coordinate MSE over the unachievable Oracle $gamma_t^*$ vanishes at rate $O(d^(-1))$—reaching $+0.02%$ excess risk at $d = 1024$ and reducing MSE by $73%–95%$ across non-stationary manifolds in $RR^(128)$.
  *(4) Sparse Exposure Mass & Rare-Token Wakeup Shock (Theorem 4):* In dense-gradient `AdamW` on Zipfian token embeddings or gated Mixture-of-Experts (MoE), global step $t$ extinguishes bias correction while dormant rows remain zero, inflicting a $(1-beta_1)/sqrt(1-beta_2) = 3.16 times – 10.0 times$ wakeup shock; per-row exposure calibration cuts rare-token SFT corruption by $8.8%–14.8%$.
  *(5) Three-Way Optimizer Asymmetry (Theorem 5):* Omitting bias correction causes under-stepping in Momentum SGD ($2.35 times$ full-batch penalty), $31.6 times$ step-1 explosion in textbook RMSprop ($+28.0%$ loss), and ratio distortion in Adam. We release open-source C\#/.NET, Python, and CUDA PyTorch (`AdaTCAdamW`) implementations validated up to $281.6"k"$-parameter GPU MoE Transformers.
]

#v(6pt)

= Introduction & Audit of Prior One-Dimensional Foundations

Exponentially Weighted Moving Averages (EWMAs) provide an $O(1)$-memory recursive mechanism for smoothing noisy streams and estimating gradient moments in deep learning @kingma2015adam @loshchilov2019decoupled @tieleman2012rmsprop. In scalar form with decay parameter $beta in [0, 1)$ and initial state $v_0 in RR$, the update
$ v_t = beta v_(t-1) + (1 - beta) x_t, quad t >= 1, $
assigns weight $(1 - beta)$ to the current observation $x_t$ and retains historical memory over an effective horizon $N_("eff") approx (1 - beta)^(-1)$. When initialized at $v_0 = 0$, early estimates are multiplicatively biased toward zero; Kingma & Ba @kingma2015adam popularized the division $hat(v)_t = v_t / (1 - beta^t)$ within the Adam optimizer.

In an earlier empirical study (`EWMABiasCorrectionResearch`) @rober2023ewma, we built an interactive simulator to examine warm-up bias, memory horizons, curve-tracking accuracy across functional families (constant, exponential, sigmoid, polynomial, sinusoidal), and qualitative links to Momentum SGD, mini-batch size, and RMSprop. Revisiting that baseline revealed both *four foundational mathematical/implementation flaws* worth correcting for the community and *a deeper open theoretical question* in higher dimensions:

1. *Discrete Step Index vs. Domain Coordinate Warp:* Passing continuous abscissae $X_i in [0, 100]$ rather than discrete step counts $t = i + 1$ into $1 - beta^t$ warps warm-up normalization whenever sample density $N != 101$ (producing errors up to $23.6$ units on standard Sigmoid curves; see Figure 1B).
2. *Non-Zero Initialization Bias ($v_0 != 0$):* Scaling $v_t / (1 - beta^t)$ leaves an uncompensated transient $(1 - beta^t)^(-1) beta^t v_0$ whenever $v_0 != 0$.
3. *Parameter Directionality & Momentum Recursion:* Larger $beta -> 1^-$ increases historical smoothing and *reduces* weight $(1 - beta)$ on recent observations (correcting inverted prose in @rober2023ewma), while standard Momentum SGD updates $bold(m)_t = beta bold(m)_(t-1) + (1 - beta) bold(g)_t$ directly from raw gradients $bold(g)_t$ without pre-adding $beta bold(m)_(t-1)$ into $bold(g)_t$.
4. *Unresolved Non-Stationary & High-Dimensional Geometry:* Even with exact zeroth-order bias correction, EWMA lags severely behind Sigmoid, Exponential, and Sinusoidal curves. Crucially, in higher dimensions ($RR^d$, $d gt.double 1$), non-stationarity interacts with anisotropic decay spectra and stochastic concentration in ways that *do not exist in $d = 1$*.

*Contributions.* This paper resolves all four issues and develops a complete operator-theoretic framework in $RR^d$. We prove exact finite-$t$ bias, covariance, and limit-duality theorems for general non-commutative contractions $bold(B) in RR^(d times d)$ (Section 2); derive an $O(1)$-per-step finite-$t$ Trend-Corrected Operator EWMA that eliminates both warm-up bias and directional manifold bending (Section 3); prove a *Blessing of Dimensionality* theorem showing that blind second-difference James–Stein shrinkage (`AdaTC-EWMA`) converges to Oracle risk at rate $O(d^(-1))$ (Section 4); and extend the theory to sparse Zipfian embeddings, MoE routers, and non-stationary neural network training via `AdaTCAdamW` (Section 5).

#figure(
  image("../figures/fig4_duality_and_bugfix.png", width: 96%),
  caption: [*(A)* Exact Theorem 1 interpolation duality from instantaneous signal $x_t$ ($beta -> 0^+$) to exact cumulative mean $macron(x)_t$ ($beta -> 1^-$). *(B)* Quantitative audit of the coordinate-vs-step warp fixed from the baseline 1D simulator ($N = 200$, $beta = 0.95$).]
)

= Operator EWMA in $RR^d$: Affine Unbiasedness & Limit Duality

Let ${bold(x)_t}_(t >= 1) subset RR^d$ satisfy $bold(x)_t = bold(mu)_t + bold(epsilon.alt)_t$ with $EE[bold(epsilon.alt)_t] = bold(0)$ and $EE[bold(epsilon.alt)_t bold(epsilon.alt)_t^top] = bold(Sigma)_t ⪰ bold(0)$. Given initial vector $bold(v)_0 in RR^d$ and contraction operator $bold(B) in RR^(d times d)$ with spectral radius $rho(bold(B)) < 1$, define:
$ bold(v)_t = bold(B) bold(v)_(t-1) + (bold(I)_d - bold(B)) bold(x)_t = bold(B)^t bold(v)_0 + sum_(k=1)^t bold(B)^(t-k)(bold(I)_d - bold(B)) bold(x)_k. $
In any matrix ring, the operator weights telescope: $sum_(k=1)^t bold(B)^(t-k)(bold(I)_d - bold(B)) = sum_(k=1)^t (bold(B)^(t-k) - bold(B)^(t-k+1)) = bold(I)_d - bold(B)^t$. Since $rho(bold(B)^t) = rho(bold(B))^t < 1$, $(bold(I)_d - bold(B)^t)$ is invertible for every $t >= 1$.

#thmbox("Theorem 1 (Affine-Invariant Operator Bias Correction, Finite-t Covariance, & Generator Duality)")[
  For any contraction $bold(B) in RR^(d times d)$ ($rho(bold(B)) < 1$) and arbitrary $bold(v)_0 in RR^d$, define the Zeroth-Order Affine-Corrected Operator EWMA:
  $ hat(bold(v))_t^((0)) := (bold(I)_d - bold(B)^t)^(-1) (bold(v)_t - bold(B)^t bold(v)_0). $
  1. *Exact Unbiasedness & Affine Equivariance:* If $EE[bold(x)_k] = bold(mu) in RR^d$ ($1 <= k <= t$), then $EE[hat(bold(v))_t^((0))] = bold(mu)$ for all $t >= 1$ and all $bold(v)_0 in RR^d$. For any $bold(A)$ commuting with $bold(B)$ and shift $bold(c) in RR^d$, mapping $bold(x)_k mapsto bold(A) bold(x)_k + bold(c)$ maps $hat(bold(v))_t^((0)) mapsto bold(A) hat(bold(v))_t^((0)) + bold(c)$.
  2. *Exact Finite-$t$ Covariance:* If $bold(epsilon.alt)_1, dots, bold(epsilon.alt)_t$ are uncorrelated with covariance $bold(Sigma)$, then
     $ bold(C)_(v,t) := "Cov"(hat(bold(v))_t^((0))) = (bold(I)_d - bold(B)^t)^(-1) [ sum_(j=0)^(t-1) bold(B)^j (bold(I)_d - bold(B)) bold(Sigma) (bold(I)_d - bold(B))^top (bold(B)^top)^j ] (bold(I)_d - bold(B)^t)^(-top), $
     which for symmetric commuting $bold(B) = bold(B)^top$ ($bold(B) bold(Sigma) = bold(Sigma) bold(B)$) reduces in $O(1)$ closed form to
     $ bold(C)_(v,t) = (bold(I)_d - bold(B))(bold(I)_d + bold(B))^(-1)(bold(I)_d + bold(B)^t)(bold(I)_d - bold(B)^t)^(-1) bold(Sigma) quad arrow.r_(t -> infinity) quad (bold(I)_d - bold(B))(bold(I)_d + bold(B))^(-1) bold(Sigma). $
  3. *Anisotropic Cumulative-Mean Duality:* Let $bold(B)(epsilon) = exp(-epsilon bold(H))$ for any matrix $bold(H) in RR^(d times d)$ whose eigenvalues have positive real parts. Then for every sample path $bold(x)_1, dots, bold(x)_t in RR^d$:
     $ lim_(bold(B) -> bold(0)) hat(bold(v))_t^((0)) = bold(x)_t, quad "and" quad lim_(epsilon -> 0^+) hat(bold(v))_t^((0))(bold(B)(epsilon)) = frac(1, t) sum_(k=1)^t bold(x)_k = macron(bold(x))_t, $
     independently of the anisotropy or non-symmetry of $bold(H)$, whereas uncorrected $lim_(epsilon -> 0^+) bold(v)_t (bold(B)(epsilon)) = bold(v)_0$.
]

#proofbox[
  Claim (1) follows immediately from $(bold(I)_d - bold(B)^t)^(-1) sum_(k=1)^t bold(B)^(t-k)(bold(I)_d - bold(B)) bold(mu) = bold(mu)$. For (2), substituting $hat(bold(v))_t^((0)) - bold(mu) = (bold(I)_d - bold(B)^t)^(-1) sum_(j=0)^(t-1) bold(B)^j (bold(I)_d - bold(B)) bold(epsilon.alt)_(t-j)$ yields the finite-horizon discrete Lyapunov sum. When $bold(B) = bold(B)^top$ commutes with $bold(Sigma)$, $sum_(j=0)^(t-1) bold(B)^(2j) = (bold(I)_d - bold(B)^(2t))(bold(I)_d - bold(B)^2)^(-1)$; factoring $bold(I)_d - bold(B)^2 = (bold(I)_d - bold(B))(bold(I)_d + bold(B))$ and $bold(I)_d - bold(B)^(2t) = (bold(I)_d - bold(B)^t)(bold(I)_d + bold(B)^t)$ cancels $(bold(I)_d - bold(B))$ and $(bold(I)_d - bold(B)^t)$ (showing $bold(C)_(v,1) = bold(Sigma)$ with zero step-1 variance blowup). For (3), expanding $exp(-epsilon bold(H)) = bold(I)_d - epsilon bold(H) + O(epsilon^2)$ gives $bold(I)_d - bold(B)(epsilon)^t = t epsilon bold(H) + O(epsilon^2)$ and $bold(B)(epsilon)^(t-k)(bold(I)_d - bold(B)(epsilon)) = epsilon bold(H) + O(epsilon^2)$. Since $"Re" lambda_i (bold(H)) > 0$ implies $det(bold(H)) != 0$, $(t epsilon bold(H))^(-1)(epsilon bold(H)) = frac(1, t) bold(I)_d$, canceling $bold(H)$ identically as $epsilon -> 0^+$.
]

= Non-Stationary Drift Lag, Directional Bending, & Exact $O(1)$ Trend Correction

Theorem 1 explains why standard bias correction succeeds on stationary plateaus yet fails on curved or drifting signals: normalizing zeroth-order mass leaves a non-vanishing first-moment delay. Crucially, in $RR^d$ ($d >= 2$), anisotropic smoothing does not merely delay a trajectory in time—it *rotates the vector direction* off the underlying manifold (Figure 2).

#thmbox("Theorem 2 (Exact Finite-t Lag Operator & First-Order Trend-Unbiased Operator EWMA)")[
  Suppose the mean trajectory evolves locally as $bold(mu)_k = bold(mu)_t - (t - k) bold(g)$ ($bold(g) in RR^d$, $1 <= k <= t$).
  1. *Exact Finite-$t$ Lag & Angular Distortion:* For any contraction $bold(B) in RR^(d times d)$ ($rho(bold(B)) < 1$),
     $ EE[hat(bold(v))_t^((0))] - bold(mu)_t = -bold(L)_t (bold(B)) bold(g), quad bold(L)_t (bold(B)) := bold(B)(bold(I)_d - bold(B))^(-1) - t (bold(I)_d - bold(B)^t)^(-1) bold(B)^t arrow.r_(t -> infinity) bold(B)(bold(I)_d - bold(B))^(-1). $
     Whenever $bold(g)$ is not an eigenvector of $bold(B)$ (e.g., multi-timescale $bold(B) = "diag"(beta_1, dots, beta_d)$ with $beta_i != beta_j$), $bold(L)_t (bold(B)) bold(g)$ is non-collinear with $bold(g)$, inducing permanent directional error in $RR^d$.
  2. *Exact $O(1)$-Time Trend-Corrected Operator EWMA (`ExactTC-EWMA`):* Augment $bold(v)_t$ with cascaded state $bold(u)_t = bold(B) bold(u)_(t-1) + (bold(I)_d - bold(B)) bold(v)_t$ from arbitrary $(bold(v)_0, bold(u)_0) in RR^d times RR^d$ (extending classical double smoothing @brown1959statistical @holt2004forecasting to exact finite-$t$ non-commutative matrix operators). Define in $O(1)$ step operations:
     $ bold(W)_t^((0))(bold(B)) := bold(I)_d - bold(B)^t - t bold(B)^t (bold(I)_d - bold(B)), quad hat(bold(u))_t^((0)) := (bold(W)_t^((0))(bold(B)))^(-1) (bold(u)_t - bold(B)^t bold(u)_0 - t bold(B)^t (bold(I)_d - bold(B)) bold(v)_0), $
     $ bold(M)_t (bold(B)) := (bold(W)_t^((0))(bold(B)))^(-1) [ 2 bold(B)(bold(I)_d - bold(B))^(-1)(bold(I)_d - bold(B)^t) - 2 t bold(B)^t - t(t-1) bold(B)^t (bold(I)_d - bold(B)) ], $
     $ bold(K)_t (bold(B)) := bold(L)_t (bold(B)) (bold(M)_t (bold(B)) - bold(L)_t (bold(B)))^(-1) quad (t >= 2; thick bold(K)_1 := bold(0)), quad lim_(t -> infinity) bold(K)_t (bold(B)) = bold(I)_d. $
     Then $hat(bold(v))_t^(("TC", 1)) := hat(bold(v))_t^((0)) + bold(K)_t (bold(B))(hat(bold(v))_t^((0)) - hat(bold(u))_t^((0)))$ satisfies $EE[hat(bold(v))_t^(("TC", 1))] equiv bold(mu)_t$ identically for every $t >= 2$, every $(bold(v)_0, bold(u)_0)$, and every contraction $bold(B)$.
]

#proofbox[
  Let $bold(S)_1 := sum_(j=0)^(t-1) j bold(B)^j (bold(I)_d - bold(B)) = (sum_(j=1)^(t-1) bold(B)^j) - (t-1) bold(B)^t = bold(B)(bold(I)_d - bold(B))^(-1)(bold(I)_d - bold(B)^t) - t bold(B)^t$. Left-multiplying by $(bold(I)_d - bold(B)^t)^(-1)$ proves (1). Unrolling $bold(u)_t$: since $(bold(I)_d - bold(B))$ commutes with powers of $bold(B)$, the coefficient of $bold(v)_0$ is $sum_(k=1)^t bold(B)^(t-k)(bold(I)_d - bold(B)) bold(B)^k = t bold(B)^t (bold(I)_d - bold(B))$, and the weight on $bold(x)_(t-j)$ is $(j+1) bold(B)^j (bold(I)_d - bold(B))^2$. Summing over $j=0..t-1$ gives $(bold(S)_1 + bold(I)_d - bold(B)^t)(bold(I)_d - bold(B)) = bold(W)_t^((0))(bold(B))$. Next, telescoping $sum_(j=0)^(t-1) j(j+1)(bold(B)^j - bold(B)^(j+1)) = 2 sum_(j=1)^(t-1) j bold(B)^j - t(t-1) bold(B)^t$ and right-multiplying by $(bold(I)_d - bold(B))$ gives $EE[hat(bold(u))_t^((0))] = bold(mu)_t - bold(M)_t (bold(B)) bold(g)$. Hence $EE[hat(bold(v))_t^((0)) - hat(bold(u))_t^((0))] = (bold(M)_t - bold(L)_t) bold(g)$, and adding $bold(K)_t (hat(bold(v))_t^((0)) - hat(bold(u))_t^((0)))$ cancels $-bold(L)_t bold(g)$ identically. As $t -> infinity$, $bold(B)^t -> bold(0)$ implies $bold(M)_t -> 2 bold(B)(bold(I)_d - bold(B))^(-1) = 2 bold(L)_infinity$, so $bold(K)_t -> bold(I)_d$.
]

#figure(
  image("../figures/fig3_anisotropic_directional_bias.png", width: 96%),
  caption: [*(A)* In $RR^(64)$ with multi-timescale $beta_i in [0.70, 0.99]$ and $bold(v)_0 != bold(0)$, standard and 0th-order bias correction retain permanent $8.5^circle$ directional rotation error, whereas `AdaTC-EWMA` reduces angular error to $0.62^circle$. *(B)* 2D slice showing off-manifold bending under anisotropic $bold(B)$.]
)

= The Blessing of Dimensionality: Blind Shrinkage (`AdaTC-EWMA`)

Unregularized trend correction ($gamma = 1$) eliminates drift lag but inflates stationary noise variance (by $approx 2.5 times$ at $beta = 0.92$). Consider the continuous shrinkage estimator for $gamma in [0, 1]$:
$ hat(bold(v))_t^(("TC", gamma)) := hat(bold(v))_t^((0)) + gamma bold(delta)_t, quad bold(delta)_t := bold(K)_t (bold(B))(hat(bold(v))_t^((0)) - hat(bold(u))_t^((0))). $
In $d = 1$, distinguishing true lag $bold(b)_t := bold(mu)_t - EE[hat(bold(v))_t^((0))]$ from stochastic noise inside $bold(delta)_t$ at a single step is fundamentally limited by $chi_1^2$ variance. In high dimensions ($d gt.double 1$), however, pooling quadratic energy across coordinates unlocks a *Blessing of Dimensionality* analogous to James–Stein shrinkage @james1961estimation.

#thmbox("Theorem 3 (Optimal Wiener/James–Stein Gate & O(1/d) Excess Risk Convergence)")[
  Let $bold(C)_(delta,t) := "Cov"(bold(delta)_t)$ and $bold(C)_(v delta, t) := "Cov"(hat(bold(v))_t^((0)), bold(delta)_t)$. For diagonal $bold(B) = "diag"(beta_1, dots, beta_d)$ and $"Cov"(bold(epsilon.alt)_k) = "diag"(sigma_1^2, dots, sigma_d^2)$, $"Tr"(bold(C)_(delta,t) + bold(C)_(v delta, t)) = sum_(i=1)^d sigma_i^2 (q_(delta,t)(beta_i) + q_(v delta,t)(beta_i))$ admits an exact $O(1)$-time polylog-geometric formula in $r_i = beta_i^2$.
  1. *Exact Oracle Gate:* $cal(R)_t (gamma) := EE[norm(hat(bold(v))_t^(("TC", gamma)) - bold(mu)_t)_2^2]$ is uniquely minimized over $[0, 1]$ at
     $ gamma_t^* = "clip"_([0,1])( frac(norm(bold(b)_t)_2^2 - "Tr"(bold(C)_(v delta, t)), norm(bold(b)_t)_2^2 + "Tr"(bold(C)_(delta,t))) ) = "clip"_([0,1])( 1 - frac("Tr"(bold(C)_(delta,t) + bold(C)_(v delta, t)), EE[norm(bold(delta)_t)_2^2]) ). $
  2. *Trend-Invariant Blind Noise Estimator:* For $t >= 3$, the second difference $Delta^2 bold(x)_t := bold(x)_t - 2 bold(x)_(t-1) + bold(x)_(t-2)$ annihilates any affine drift $bold(mu)_0 + t bold(g)$ identically ($Delta^2 bold(mu)_t equiv bold(0)$) while satisfying $EE[(Delta^2 x_(t,i))^2 / 6] = sigma_i^2$.
  3. *$O(d^(-1))$ Oracle Convergence:* Under independent sub-Gaussian noise with bounded fourth moments and interior optimum $gamma_t^* in (0, 1)$, the fully blind plug-in gate
     $ hat(gamma)_t := "clip"_([0,1])( 1 - frac(sum_(i=1)^d hat(sigma)_(t,i)^2 (q_(delta,t)(beta_i) + q_(v delta,t)(beta_i)), norm(bold(delta)_t)_2^2) ) $
     satisfies $|hat(gamma)_t - gamma_t^*| = O_PP (d^(-1/2))$ and achieves quadratic excess per-coordinate risk:
     $ frac(1, d) EE[norm(hat(bold(v))_t^(("TC", hat(gamma)_t)) - bold(mu)_t)_2^2] - frac(1, d) cal(R)_t (gamma_t^*) = O(d^(-1)). $
]

#proofbox[
  Writing $hat(bold(v))_t^((0)) - bold(mu)_t = -bold(b)_t + bold(xi)_t$ and $bold(delta)_t = bold(b)_t + bold(eta)_t$ gives $cal(R)_t (gamma) = (1 - gamma)^2 norm(bold(b)_t)_2^2 + "Tr"(bold(C)_(v,t)) + 2 gamma "Tr"(bold(C)_(v delta,t)) + gamma^2 "Tr"(bold(C)_(delta,t))$. Differentiating and using $EE[norm(bold(delta)_t)_2^2] = norm(bold(b)_t)_2^2 + "Tr"(bold(C)_(delta,t))$ yields $gamma_t^*$. By the Hanson–Wright inequality @rudelson2013hanson, $frac(1, d) norm(bold(delta)_t)_2^2$ and $frac(1, d) sum_i hat(sigma)_(t,i)^2 (q_(delta,t) + q_(v delta,t))$ concentrate around their means at rate $O_PP (d^(-1/2))$, whence $EE[(hat(gamma)_t - gamma_t^*)^2] = O(d^(-1))$. Since $cal(R)_t ' (gamma_t^*) = 0$ when $gamma_t^* in (0, 1)$, exact second-order Taylor expansion gives $frac(1, d)(cal(R)_t (hat(gamma)_t) - cal(R)_t (gamma_t^*)) = frac(norm(bold(b)_t)_2^2 + "Tr"(bold(C)_(delta,t)), d)(hat(gamma)_t - gamma_t^*)^2 = O(d^(-1))$.
]

#figure(
  image("../figures/fig2_blessing_of_dimensionality.png", width: 96%),
  caption: [*(A)* Blind `AdaTC-EWMA` (unknown $sigma^2$) strictly dominates both Standard Bias Correction ($gamma=0$) and Full Trend Correction ($gamma=1$) for all $d >= 4$. *(B)* Empirical excess risk over the unachievable Oracle $gamma_t^*$ follows the exact theoretical $O(d^(-1))$ slope across ten octaves ($+0.02%$ excess at $d = 1024$).]
)

= Deep Learning Optimizers: Sparse Exposure & Batch-Size Asymmetry

Extending Appendices A–C of @rober2023ewma to high-dimensional neural optimization exposes two critical structural phenomena in modern training pipelines (`AdamW`, `RMSprop`, and `Momentum SGD`).

#thmbox("Theorem 4 (Sparse Exposure Mass & The Rare-Token Wakeup Shock in Dense AdamW)")[
  For time-varying diagonal contractions $bold(B)_t = "diag"(beta_(t,1), dots, beta_(t,d))$, the exact stationary-unbiased estimator is $hat(bold(v))_t = "diag"(bold(w)_t)^(-1)(bold(v)_t - (bold(1)_d - bold(w)_t) ⊙ bold(v)_0)$ where $bold(w)_t = bold(B)_t bold(w)_(t-1) + (bold(I)_d - bold(B)_t) bold(1)_d$. In standard dense-gradient `AdamW` (e.g., `nn.Embedding(sparse=False)` or gated MoE experts), a parameter row $r$ remaining inactive ($bold(g)_(k,r) = bold(0)$) until global step $t gt.double (1 - beta_2)^(-1)$ experiences $1 - beta_1^t approx 1$ and $1 - beta_2^t approx 1$, receiving an uncalibrated first-activation step shock:
  $ kappa_t (beta_1, beta_2) = frac(1 - beta_1, 1 - beta_1^t) sqrt(frac(1 - beta_2^t, 1 - beta_2)) arrow.r_(t -> infinity) frac(1 - beta_1, sqrt(1 - beta_2)) = cases(3.162 times & "for " (beta_1, beta_2) = (0.9, 0.999), 10.00 times & "for " (beta_1, beta_2) = (0.9, 0.9999)). $
  Maintaining lazy per-row step clocks or exposure masses $(bold(w)_(1,t), bold(w)_(2,t))$ restores exact $1.00 times$ step calibration at all activation frequencies.
]

#thmbox("Theorem 5 (Three-Way Bias-Correction Asymmetry Across Momentum, RMSprop, & Adam)")[
  Omitting bias correction scales iteration $t$'s parameter update by: *(i)* $(1 - beta_1^t) < 1$ in *Momentum SGD* (cold-start damping, causing severe loss degradation when batch size $B$ is large so step budget $T = E N / B$ is small); *(ii)* $(1 - beta_2^t)^(-1/2) > 1$ in textbook *RMSprop* ($31.62 times$ step explosion at $t=1$ for $beta_2 = 0.999$); and *(iii)* $(1 - beta_1^t)/sqrt(1 - beta_2^t)$ in *Adam* ($3.162 times$ step-1 overshoot for $beta_1=0.9, beta_2=0.999$).
]

Combining Theorems 1–4 yields `AdaTCAdamW` (implemented as a drop-in `torch.optim.Optimizer`), which pairs blind `AdaTC` trend-corrected first moments on dense parameter tensors with innovation-triggered second-moment shift damping ($norm(bold(g)_t)_2^2 / norm(hat(bold(v))_(t-1))_1 > tau_("shift")$) and row-wise lazy exposure calibration on sparse embedding/MoE tensors.

= Comprehensive Empirical Evaluation

#figure(
  image("../figures/fig1_curve_families_hd.png", width: 98%),
  caption: [Representative coordinate trajectories in $RR^(128)$ ($beta = 0.92$, 50 Monte Carlo trials per family). In Panel 6, the blind shrinkage gate $hat(gamma)_t$ (amber) automatically drops near $0$ on flat noisy plateaus and rises to $1$ on steep ramps.]
)

#figure(
  table(
    columns: (2.1fr, 0.9fr, 1.0fr, 1.0fr, 1.1fr, 0.95fr, 0.95fr),
    inset: 5.5pt,
    align: (left, right, right, right, right, right, right),
    stroke: 0.5pt + rgb("#cbd5e1"),
    fill: (_, row) => if row == 0 { rgb("#eff6ff") } else { none },
    [*Trajectory Family ($RR^(128)$)*], [*Uncorr.*], [*Std BC*], [*Exact TC*], [*AdaTC (Ours)*], [*vs. Std BC*], [*vs. Exact TC*],
    [Constant + Noise *(Control)*], [504.91], [*9.72*], [24.72], [*9.91*], [-1.9%], [*+59.9%*],
    [Sigmoid Transition], [146.00], [142.24], [17.28], [*16.11*], [*+88.7%*], [*+6.8%*],
    [Exponential Drift], [974.69], [783.35], [36.74], [*35.67*], [*+95.4%*], [*+2.9%*],
    [Polynomial Curve], [76.81], [62.41], [9.21], [*7.69*], [*+87.7%*], [*+16.4%*],
    [Sinusoidal Manifold], [577.19], [399.11], [108.26], [*107.14*], [*+73.2%*], [*+1.0%*],
    [Alternating Plateaus & Ramps], [365.70], [280.83], [62.56], [*61.29*], [*+78.2%*], [*+2.0%*],
  ),
  caption: [Time-averaged per-coordinate MSE across 50 Monte Carlo runs in $RR^(128)$ ($beta = 0.92$, 100% blind noise estimation). Paired $t$-tests confirm $p < 10^(-12)$ across all five non-stationary families.]
)

*1. Non-Stationary Manifolds & Dimensionality Scaling.* Table 1 and Figures 2–3 confirm all predictions of Theorems 1–3. Across Sigmoid, Exponential, Polynomial, Sinusoidal, and Regime-Switching streams in $RR^(128)$, blind `AdaTC-EWMA` reduces MSE by *73.2% to 95.4%* over Standard Bias Correction ($p < 10^(-12)$), while avoiding the $2.54 times$ stationary noise penalty of unregularized `ExactTC` on Constant + Noise (`9.91` vs. `24.72`, a *+59.9%* improvement). Across dimensions $d in {2, 4, 8, dots, 1024}$, excess risk halves every time $d$ doubles: `+13.93%` ($d=2$) $->$ `+3.28%` ($d=8$) $->$ `+0.81%` ($d=32$) $->$ `+0.18%` ($d=128$) $->$ *`+0.02%`* ($d=1024$).

#figure(
  image("../figures/fig5_deep_learning_optimizers.png", width: 98%),
  caption: [*(A)* Zipfian SFT rare-token embedding corruption ($d = 64"k"$): Theorem 4 eliminates the wakeup shock, cutting tail MSE by $+8.8%$ ($beta_2 = 0.999$) and $+14.8%$ ($beta_2 = 0.9999$). *(B)* Non-stationary 2-layer MLP ($d = 17.7"k"$) with regime jump at $t = 250$. *(C)* Batch-size asymmetry confirming Theorem 5.]
)

*2. High-Dimensional Neural Networks & CUDA GPU Validation.* Figure 5 summarizes three multi-seed deep learning experiments:
- *Zipfian Embedding Fine-Tuning ($V=2000, h=32 => d = 64,000$):* Standard global-$t$ `AdamW` inflicts an empirical $2.02 times – 2.29 times$ wakeup shock on tail tokens ($v > 400$), degrading pretrained embeddings; `SparseMass-AdamW` reduces visited tail MSE by *+8.8%* ($beta_2=0.999$) and *+14.8%* ($beta_2=0.9999$).
- *Non-Stationary ReLU MLP ($d = 17,668$, 5 seeds):* Adding bias correction to `RMSprop` ($beta_2 = 0.999$) eliminates the $31.6 times$ step-1 loss spike and lowers overall loss by *+28.0%* (`0.8355` $->$ `0.6013`). `AdaTC-AdamW` outperforms standard `AdamW` across overall loss (`0.6206` vs. `0.6484`), post-shift adaptation (`1.4549` vs. `1.5622`, *+6.9%*), and final loss (`0.6406` vs. `0.6559`).
- *Batch-Size Scaling ($B in {8, dots, 2048}$):* Ignoring bias correction in Momentum SGD incurs $1.00 times$ loss at $B = 8$ ($2560$ steps) versus *$2.35 times$ higher loss* at full batch $B = 2048$ ($10$ steps), quantitatively proving Theorem 5.
- *CUDA GPU MoE Transformer ($d = 281,608$ on NVIDIA RTX 5060, 3 seeds):* Training a Multi-Head Self-Attention + 8-expert Top-1 Gated MoE sequence model under mid-training rule shift ($t=150$), drop-in `AdaTCAdamW` consistently improves overall cross-entropy (`6.0871` vs. `6.0938`), post-shift recovery (`6.5305` vs. `6.5468`), and final convergence (`5.7458` vs. `5.7477`) over `torch.optim.AdamW`.

= Conclusion

By lifting EWMA bias correction from scalar stationarity to contraction operators in $RR^d$, we showed that initialization bias, non-stationary lag, anisotropic vector rotation, and sparse-token wakeup shocks admit unified $O(1)$-per-step closed-form remedies—and that high dimensionality *helps rather than hurts* adaptive trend correction via $O(d^(-1))$ concentration. Full source code in C\#/.NET, Python, and PyTorch is openly available at `https://github.com/AndrewRober/HigherDimensionalEWMABiasCorrection`.

#bibliography("references.bib", title: "References", style: "ieee")
