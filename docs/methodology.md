# Methodology

This note collects the derivations behind `dutchbook`. Notation: there are $n$ atomic
events and $k$ distinct *worlds* (truth assignments, merged when they agree on every
formula), $m$ formulas $\varphi_1,\dots,\varphi_m$, and the truth matrix
$W\in\{0,1\}^{k\times m}$ with $W_{wj}=1$ iff $\varphi_j$ is true in world $w$. A forecaster
quotes probabilities $q\in[0,1]^m$; $q_j$ is the price of a contract that pays $1$ if
$\varphi_j$ is true. $\Delta$ is the probability simplex over worlds.

## 1. Coherence and Dutch books

The **coherent polytope** is the set of probabilities that some probability measure over the
worlds induces,

$$
C=\{W^\top\lambda:\ \lambda\in\Delta\}=\operatorname{conv}\{W_{w\cdot}\}.
$$

A bettor chooses stakes $s\in\mathbb R^m$ (positive: buy the contract from the forecaster at
price $q_j$; negative: sell it) and earns

$$
\pi_w(s)=\sum_j s_j\,(W_{wj}-q_j)\qquad\text{in world }w .
$$

**Theorem (de Finetti 1937).** $q\in C$ if and only if there is no $s$ with $\pi_w(s)>0$ for
every world $w$.

*Proof.* If $q=W^\top\lambda$ then $\sum_w\lambda_w\pi_w(s)=0$ for every $s$, so some
$\pi_w(s)\le0$. If $q\notin C$, the separating hyperplane theorem gives $s$ with
$s\cdot p<s\cdot q$ for all $p\in C$ (in particular for every row of $W$), which is
$\pi_w(s)>0$ for every $w$. $\square$

### Pricing the book: LP duality

Positive profit can be scaled without bound, so the stakes are normalised in a norm. The
guaranteed profit per unit of stake norm is

$$
V_{\|\cdot\|}(q)=\max_{\|s\|\le1}\ \min_{w}\ \pi_w(s)
=\max_{\|s\|\le1}\ \min_{\lambda\in\Delta}\ s\cdot(W^\top\lambda-q).
$$

The objective is bilinear in $(s,\lambda)$ and both sets are compact and convex, so the two
optimisations can be exchanged (von Neumann's minimax theorem):

$$
V_{\|\cdot\|}(q)=\min_{\lambda\in\Delta}\ \max_{\|s\|\le1}\ s\cdot(W^\top\lambda-q)
=\min_{p\in C}\ \|p-q\|_{*},
$$

the distance from $q$ to the coherent polytope in the **dual norm**. Three choices are
implemented:

- total stake $\|s\|_1\le1$ (as in Andrews and Sarkar, 2026): profit equals the sup-norm
  distance $\min_{p\in C}\|p-q\|_\infty$;
- each stake $\|s\|_\infty\le1$: profit equals the $\ell_1$ distance;
- Euclidean stakes $\|s\|_2\le1$: profit equals the $\ell_2$ distance.

For the first two, `find_dutch_book` solves the bettor's LP directly (variables $s$ or
$s=u-v$, one constraint per distinct world, HiGHS) and `coherence_distance` solves the
distance LP in $\lambda$; the two are independent formulations whose optimal values agree to
machine precision. For the Euclidean norm the optimal stakes are the normalised projection
residual. With $p^{\ast}$ the Euclidean projection of $q$ onto $C$ and $d=\|p^{\ast}-q\|_2$,

$$
s^{\ast}=\frac{p^{\ast}-q}{d},\qquad
\pi_w(s^{\ast})=\frac{(p^{\ast}-q)\cdot(W_{w\cdot}-q)}{d}
=d+\frac{(p^{\ast}-q)\cdot(W_{w\cdot}-p^{\ast})}{d}\ \ge\ d ,
$$

because $(q-p^{\ast})\cdot(v-p^{\ast})\le0$ for every $v\in C$ (the variational inequality that
characterises the projection).

The optimal LP multipliers on the world constraints form a probability measure $\lambda^{\ast}$
over worlds with $\|W^\top\lambda^{\ast}-q\|_*=V(q)$: the coherent forecast closest to $q$.

### Elementary invariances (tested)

The guaranteed profit is unchanged by permuting formulas or worlds, and by replacing every
formula by its negation together with $q\mapsto 1-q$ (then $W\mapsto 1-W$). It is zero
exactly on $C$.

## 2. Coherentization

### Euclidean (Brier) projection

$p^{\ast}=\arg\min_{p\in C}\|p-q\|_2^2$ is computed with **Wolfe's min-norm-point algorithm**
(Wolfe 1976) applied to $C-q$. It needs only a linear minimisation oracle
$g\mapsto\arg\min_{v\in\text{vertices}}\langle g,v\rangle$. Each major cycle asks the oracle
for the vertex minimising $\langle x,v-q\rangle$ at the current point $x=p-q$ and stops when
the Frank-Wolfe gap

$$
\mathrm{gap}=2\Bigl(\|x\|^2-\min_v\langle x,v-q\rangle\Bigr)\ \le\ \text{tol}\qquad(\text{default }10^{-12}).
$$

Otherwise the vertex joins the *corral*; minor cycles move to the minimum-norm point of the
affine hull of the corral (the bordered system $\bigl[\begin{smallmatrix}AA^\top&1\\1^\top&0\end{smallmatrix}\bigr]$,
non-singular exactly when the corral is affinely independent) and drop vertices whose
barycentric weight reaches zero. The algorithm terminates finitely in exact arithmetic. The
gap bounds the suboptimality, $f(p)-f^{\ast}\le\mathrm{gap}$ for $f=\|p-q\|^2$.

### KL (log score) projection

$p^{\ast}=\arg\min_{p\in C}\sum_j \mathrm{KL}(p_j\|q_j)$ with binary Kullback-Leibler divergences.
Because the log score of an exact $0$ or $1$ is infinite, $q$ is clipped to
$[\varepsilon,1-\varepsilon]$ ($\varepsilon=10^{-6}$). The objective is convex and smooth in
the relative interior, with a barrier at the boundary. The solver is **away-step
Frank-Wolfe** with exact line search: the directional derivative of $f$ along a segment is
monotone, so the line minimiser is the root of one scalar equation. It starts at the
barycentre of the extreme vertices along every coordinate (finite gradient), and stops when
the Frank-Wolfe gap $\langle\nabla f(p),p-s\rangle$ is at most $10^{-10}$; this is an upper
bound on $f(p)-f^{\ast}$ for any convex $f$.

### The accuracy-dominance certificate

Let $D$ be the Bregman divergence of an additive strictly proper scoring rule: the squared
Euclidean distance for the Brier score $S(y,w)=\sum_j (y_j-W_{wj})^2$, and the sum of binary
KL divergences for the log score. In both cases the score of a forecast in a world is the
divergence from the world's truth vector, $S(y,w)=D(W_{w\cdot},y)$ (Savage 1971; Predd et al.
2009). The **three-point identity** holds for every Bregman divergence:

$$
D(v,q)-D(v,p)-D(p,q)=\bigl\langle\nabla_pD(p,q),\,v-p\bigr\rangle .
$$

At the minimiser $p^{\ast}$ of $D(\cdot,q)$ over the convex set $C$ the right-hand side is
non-negative for every $v\in C$ (first-order optimality). Taking $v=W_{w\cdot}$:

$$
S(q,w)-S(p^{\ast},w)\ \ge\ D(p^{\ast},q)\qquad\text{for every world }w .
$$

For the Brier score this reads $\mathrm{Brier}(q,w)-\mathrm{Brier}(p^{\ast},w)\ge\|p^{\ast}-q\|^2$
(Joyce 1998; Predd et al. 2009). If $q$ is incoherent then $D(p^{\ast},q)>0$, so the repaired
vector is strictly more accurate **in every world**, whatever happens: the repair is free in
the sense that it uses no outcome data.

With an approximate minimiser $p$ the margin
$S(q,w)-S(p,w)-D(p,q)=\langle\nabla_pD(p,q),W_{w\cdot}-p\rangle$ is at least $-\mathrm{gap}$,
because $\min_{v}\langle\nabla f(p),v-p\rangle=-\mathrm{gap}$ is exactly the Frank-Wolfe
quantity the solver drives to zero. `dominance_certificate` therefore checks
$\text{margin}\ge-\mathrm{gap}-10^{-9}$ in every distinct world; on enumerated problems the
check covers all worlds, and with column generation the statement holds for all worlds by the
same inequality because the gap is computed against the MILP oracle.

**The guarantee concerns the whole vector.** It does not say that every sub-block improves.
Example: $q=(0.9,0.9)$ for the formulas $A,\neg A$ projects to $(0.5,0.5)$. The total Brier
score falls from $0.82$ to $0.50$ if $A$ is true, but the first coordinate alone worsens from
$(0.9-1)^2=0.01$ to $(0.5-1)^2=0.25$. The same happens on ForecastBench: the Brier score of
the four joint cells alone is not guaranteed to fall (it does so on average, and `results.json`
reports the share of tuples where it rises).

## 3. Coherent extension bounds

For coherent $q$ on $\varphi_1,\dots,\varphi_m$ and a further formula $\psi$ with truth vector
$t\in\{0,1\}^k$ over the worlds, de Finetti's fundamental theorem of prevision says the
values of $P(\psi)$ that keep the enlarged system coherent form the interval
$[\min_\lambda t^\top\lambda,\ \max_\lambda t^\top\lambda]$ over $\{\lambda\in\Delta:\ W^\top\lambda=q\}$, two
linear programs. Examples checked in the tests:

- Modus ponens. With $P(A)=a$ and $P(A\wedge B)=a\,P(B\mid A)$, the interval for $B$ is
  $[\,a\,P(B\mid A),\ a\,P(B\mid A)+1-a\,]$.
- Fréchet bounds. $\max(0,a+b-1)\le P(A\wedge B)\le\min(a,b)$ and
  $\max(a,b)\le P(A\vee B)\le\min(1,a+b)$.

## 4. Column generation for many atoms

Enumeration is limited to 22 atoms. Beyond that, `find_dutch_book_cg` keeps a restricted
master LP over a growing set of worlds. With master stakes $s$, the **pricing problem**
$\min_w\,s\cdot(W_{w\cdot}-q)$ is a mixed-integer program: one binary variable per atom and
per subformula, linked by the standard encodings

$$
y=l\wedge r:\ y\le l,\ y\le r,\ y\ge l+r-1;\qquad
y=l\vee r:\ y\ge l,\ y\ge r,\ y\le l+r;
$$
$$
y=\neg a:\ y=1-a;\qquad
y=(l\leftrightarrow r):\ y\le1\mp(l-r),\ y\ge l+r-1,\ y\ge1-l-r .
$$

The master value is an **upper** bound on the guaranteed profit (fewer worlds); the pricing
value $\min_w s\cdot(W_{w\cdot}-q)$ is a rigorous **lower** bound (the master stakes are
feasible for the full problem and the oracle minimises over all worlds). The loop stops when
they meet, so the reported profit is certified in every world. The same oracle plugs into the
Wolfe and Frank-Wolfe projections. Tests cross-check against enumeration on random formula
sets up to 16 atoms.

## 5. Pairwise question graphs: local and global coherence

Take questions as nodes and, for every pair, the four joint cells as an edge, with one marginal
per node shared by all its edges. *Local* coherence audits each pair (two marginals and four
cells) separately; *global* coherence audits the whole system against one joint distribution.

- On a **forest** local coherence implies global coherence: pairwise tables that agree on
  their shared marginals glue along a tree into one joint distribution.
- With a **cycle** it fails: three events with all three pairs "always different" are
  pairwise coherent, but an odd cycle cannot be properly 2-coloured, so a Dutch book exists.

**Experiment on ForecastBench graphs.** For one model configuration, round and resolution date,
the tuples form such a graph (questions as nodes, tuples as edges, marginals shared). To isolate
global effects every edge table is replaced by the closest joint table with the model's own
marginals: the joint tables with marginals $a,b$ are the one-parameter family
$(t,\ a-t,\ b-t,\ 1-a-b+t)$ with $t\in[\max(0,a+b-1),\min(a,b)]$, and the Euclidean distance
to the elicited cells is minimised in closed form and clipped to that interval. Each pair is then
coherent and the pairs agree on marginals, so incoherence of a tree is impossible and any
incoherence of a component with a cycle is global. Components of at most 10 questions are
enumerated, larger ones (up to 30, a runtime cap recorded in `results.json`) use column
generation.

A book against one pair is a book against the whole system (same stakes, zeros elsewhere), so
the global profit is never below the largest local profit. Components of the graph are
independent and the sup-norm distance to a product of polytopes is the maximum of the
component distances.

## 6. The ForecastBench audit

**Data.** The processed forecast sets of ForecastBench (Karger et al. 2025), CC BY-SA 4.0,
downloaded at run time into `data/raw/` and pinned in `docs/results.json` by URL, ETag, size,
SHA-256 and download date. Rounds up to 2025-08-31 contain *combination questions*: for a
pair of binary questions $(A,B)$ and a resolution date, the four joint cells
$A\wedge B,\ A\wedge\neg B,\ \neg A\wedge B,\ \neg A\wedge\neg B$, each a separate forecast.
Together with the forecasts for $A$ and $B$ themselves (the same questions and resolution
date) this gives the six formulas $A,\ B,\ A\wedge B,\ A\wedge\neg B,\ \neg A\wedge B,\ \neg A\wedge\neg B$
over two atoms, four worlds.

**Tuple construction.** A tuple needs all four cells and both marginals, none imputed by the
benchmark (imputed values are a default filled in for a failed model response). Tuples with
missing marginals or incomplete cells are dropped; the counts are in `results.json` under
`extraction`. A tuple is *resolved* if all six events are resolved to $0$ or $1$; the realised
world is then the cell resolved to $1$, and the marginals' resolutions are consistent with it
for every resolved tuple.

**Model configurations.** The unit is a *configuration*: model plus prompt variant (zero shot,
scratchpad, with news, with freeze values, and so on), as named by the benchmark. One
configuration, `claude-3-5-haiku-20241022`, has joint cells equal to the products of the
marginals in every tuple (derived under independence rather than elicited) and is excluded
from model summaries because its coherence holds by construction. The benchmark's own
constant, random and naive baselines are reported separately as references.

**Metrics per tuple.**

1. *Guaranteed arbitrage per unit of total stake*: the sup-norm distance to the coherent
   polytope, an exact LP.
2. *Renormalisation baseline*: rescale the four joint cells to sum to one, keep the marginals,
   and recompute. This removes the part of the incoherence that arises mechanically when each
   cell is forecast in its own prompt. Because the marginals were also elicited in separate
   prompts, what remains is robust to the sum-to-one artefact but not to every elicitation
   artefact.
3. *Brier projection* and the per-world dominance margins in all four worlds, for every tuple
   whether resolved or not.
4. On resolved tuples, the Brier score (mean squared error over the six coordinates) as
   elicited, after the Brier projection and after renormalisation, and the same for the
   four joint cells alone.

**Materiality.** Forecasts carry at most a few decimals; a forecast vector that is coherent
before rounding to $d$ decimals lies within $0.5\cdot10^{-d}$ of the polytope in the sup norm,
so almost every tuple is formally incoherent. The headline therefore also reports the share of
tuples with guaranteed profit of at least 1%, 5% and 10% per unit of stake.

**Uncertainty.** Pooled statistics carry 95% percentile intervals from a bootstrap over
question pairs (1,000 resamples, seed in `results.json`), because the same pair is forecast by
many configurations and in several rounds. Model configurations are not weighted: configurations
with more tuples count more in pooled numbers, and per-configuration rows are in `results.json`.

## 7. Validation

`scripts/validation_experiments.py` reruns the invariants on 2,000 seeded random instances and
writes the extremes to `results.json`; the test suite checks the same invariants on smaller
samples and adds textbook cases (a forecaster who prices $A$ and $\neg A$ at $0.6$ each, the
Linda conjunction fallacy, modus ponens bounds).

- LP strong duality for all three stake norms: independent primal and dual LP, and a
  nonlinear solver.
- Euclidean projection: exact support enumeration of the quadratic program.
- Projection optimality: the variational inequality over every world and the Frank-Wolfe gap.
- Brier and log dominance: per-world margin at least $-\mathrm{gap}$ on random instances and on
  all ForecastBench tuples.
- KL projection: first-order optimality over every world and a generic nonlinear solver.
- Extension bounds: modus ponens and Fréchet closed forms.
- Column generation: enumeration on up to 16 atoms.

## References

- Andrews, I. and Sarkar, S. (2026). *Dutch Books for Language Models*. arXiv:2609.02797.
- de Finetti, B. (1937). La prévision: ses lois logiques, ses sources subjectives. *Annales de
  l'Institut Henri Poincaré* 7, 1-68.
- Joyce, J. M. (1998). A nonpragmatic vindication of probabilism. *Philosophy of Science* 65(4),
  575-603.
- Karger, E., Bastani, H., Yueh-Han, C., Jacobs, Z., Halawi, D., Zhang, F. and Tetlock, P. E.
  (2025). ForecastBench: A dynamic benchmark of AI forecasting capabilities. *ICLR 2025*.
- Paleka, D., Sudhir, A. P., Alvarez, A., Bhat, V., Shen, A., Wang, E. and Tramèr, F. (2024).
  *Consistency Checks for Language Model Forecasters*. arXiv:2412.18544.
- Predd, J. B., Seiringer, R., Lieb, E. H., Osherson, D. N., Poor, H. V. and Kulkarni, S. R.
  (2009). Probabilistic coherence and proper scoring rules. *IEEE Transactions on Information
  Theory* 55(10), 4786-4792.
- Savage, L. J. (1971). Elicitation of personal probabilities and expectations. *Journal of the
  American Statistical Association* 66(336), 783-801.
- Wolfe, P. (1976). Finding the nearest point in a polytope. *Mathematical Programming* 11,
  128-149.
