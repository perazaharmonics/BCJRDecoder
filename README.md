# BCJR Decoder

A **Max-Log-MAP** (Bahl-Cocke-Jelinek-Raviv) soft-input/soft-output decoder for the **CCSDS 131.0-B-5** standard *K = 7, rate-1/2* inner convolutional code. The decoder runs forward/backward recursions over the 64-state trellis and emits both **hard decisions** and **a-posteriori (APP) log-likelihood ratios** per information bit.

---

## 1. Overview

The convolutional code is defined by the octal generator polynomials

$$
g_1 = 171_8 = \mathtt{1111001_2}, \qquad
g_2 = 133_8 = \mathtt{1011011_2},
$$

with constraint length $K = 7$, giving a shift register of $K-1 = 6$ memory bits and therefore

$$
N_{\text{states}} = 2^{K-1} = 2^{6} = 64 .
$$

Per CCSDS 131.0-B-5 §3.3.1(5), the $G_2$ output is **inverted** at the encoder; the trellis is built to expect the inverted $C_2$ so the decoder mirrors the encoder exactly.

### LLR convention

All log-likelihood ratios use the **bit-0-favoring** convention:

$$
L \;=\; \log \frac{P(\text{bit} = 0)}{P(\text{bit} = 1)},
\qquad
L > 0 \Rightarrow \text{bit} = 0 .
$$

Coded bits are mapped to BPSK-style antipodal symbols:

$$
x \;=\; (-1)^{c} \;=\;
\begin{cases}
+1, & c = 0,\\[2pt]
-1, & c = 1.
\end{cases}
$$

---

## 2. Trellis Model

At information step $t$ the shift register is in state $s_t \in \{0, \dots, 63\}$. An input bit $b \in \{0, 1\}$ drives the transition

$$
z'_t = \big( (s_t \ll 1) \mathbin{|} b \big) \mathbin{\&} \mathtt{0x7F},
\qquad
s_{t+1} = z'_t \mathbin{\&} \mathtt{0x3F},
$$

and produces the two expected coded bits via parity over the generator masks:

$$
c_1 = \mathcal{P}\big( z'_t \mathbin{\&} g_1 \big),
\qquad
c_2 = \mathcal{P}\big( z'_t \mathbin{\&} g_2 \big),
$$

where $\mathcal{P}(\cdot)$ is the **even-parity** (XOR-fold) operator

$$
\mathcal{P}(w) \;=\; \bigoplus_{i} w_i \;=\; \left( \sum_i w_i \right) \bmod 2 .
$$

Each transition $(s_t, b) \to (s_{t+1}, \mathbf{c})$ is stored once at trellis-build time.
---

## 3. Branch Metric (Gamma)

For a transition at step $t$ from state $s$ with input bit $b$, the expected coded symbols are $x_1, x_2 \in \{+1,-1\}$ and the channel LLRs are $L_{c_1}[t], L_{c_2}[t]$. The (log-domain) branch metric is

$$
\gamma_t(s,b)
\;=\;
\tfrac{1}{2}\Big( x_1\, L_{c_1}[t] \;+\; x_2\, L_{c_2}[t] \Big).
$$

A flat a-priori on the information bit is assumed, $L_a(u) = 0$. **Punctured / erased** symbols carry no information and are passed as $L_{c_i}[t] = 0$.

---

## 4. Forward and Backward Recursions

Let $\alpha_t(s)$ and $\beta_t(s)$ be the forward and backward state metrics. Under the **Max-Log** approximation

$$
\log \sum_i e^{x_i} \;\approx\; \max_i x_i ,
$$

the recursions reduce to additions and maximizations.

### Forward recursion ($t = 0 \to N$)

$$
\alpha_{t+1}(s')
\;=\;
\max_{(s,b)\,:\, s \to s'}
\Big[\, \alpha_t(s) + \gamma_t(s,b) \,\Big],
$$

with the **known start state** boundary condition

$$
\alpha_0(s) =
\begin{cases}
0, & s = 0,\\
-\infty, & s \neq 0.
\end{cases}
$$

### Backward recursion ($t = N \to 0$)

$$
\beta_t(s)
\;=\;
\max_{b \in \{0,1\}}
\Big[\, \beta_{t+1}\big(\eta(s,b)\big) + \gamma_t(s,b) \,\Big],
$$

where $\eta(s,b)$ is the next state. The endpoint is **free (non-terminated)**, so

$$
\beta_N(s) = 0 \quad \forall\, s .
$$

In code, $-\infty$ is represented by `negeps` $= -10^{30}$.

---

## 5. A-Posteriori LLR and Hard Decision

For each information bit $u_t$, collect the best path metric over all transitions consistent with $u_t = 0$ and $u_t = 1$:

$$
m_0[t] = \max_{(s,b)\,:\, b = 0}
\Big[\alpha_t(s) + \gamma_t(s,0) + \beta_{t+1}\big(\eta(s,0)\big)\Big],
$$

$$
m_1[t] = \max_{(s,b)\,:\, b = 1}
\Big[\alpha_t(s) + \gamma_t(s,1) + \beta_{t+1}\big(\eta(s,1)\big)\Big].
$$

The **a-posteriori LLR** is their difference:

$$
L(u_t)
\;=\;
\log \frac{P(u_t = 0 \mid \mathbf{y})}{P(u_t = 1 \mid \mathbf{y})}
\;\approx\;
m_0[t] - m_1[t].
$$

The **hard decision** follows directly:

$$
\hat{u}_t =
\begin{cases}
0, & L(u_t) \ge 0,\\
1, & L(u_t) < 0.
\end{cases}
$$

---

## 6. Complexity

For a block of $N$ information bits over $S = 64$ states with $2$ branches per state:

$$
\mathcal{O}\big(N \cdot S \cdot 2\big) \;=\; \mathcal{O}(128\,N)
$$

time, and

$$
\mathcal{O}\big((N+1)\cdot S\big)
$$

memory for the $\alpha$ and $\beta$ trellises. All operations are additions and comparisons ? there are **no exponentials or transcendental functions** in the Max-Log path.

---

## 7. API

```cpp
#include "BCJRDecoder.hpp"

BCJRDecoder dec;                 // builds the 64-state trellis in the ctor

std::vector<uint8_t> bits;       // hard decisions out
std::vector<float>   app;        // a-posteriori LLRs out

BCJRDecoder::LLR l;
l.lc1 = /* channel LLRs for C1, one per info step */;
l.lc2 = /* channel LLRs for C2 (pre-flipped for G2 inversion) */;
l.out = &bits;
l.llr = &app;

dec.Decode(l);                   // fills bits[] and app[]
```

### Input contract

| Field | Meaning |
|-------|---------|
| `lc1[t]` | Channel LLR of coded bit $C_1$ at step $t$ |
| `lc2[t]` | Channel LLR of coded bit $C_2$ at step $t$ ? **sign-flipped** to account for the CCSDS $G_2$ inversion |
| `out`    | Pointer to output hard-decision buffer ($0/1$) |
| `llr`    | Pointer to output APP LLR buffer ($L > 0 \Rightarrow \text{bit }0$) |

> **Puncturing / erasures:** pass $L_{c_i}[t] = 0$ for any symbol that was punctured or lost; the branch metric then contributes no information for that coded bit.

---

## 8. Use Cases

- **CCSDS telemetry reception** ? soft-decision decoding of the standard $K=7$, rate-1/2 inner code in deep-space and near-Earth links.
- **Turbo / iterative decoding** ? the APP LLR output $L(u_t)$ serves as **extrinsic information** for an outer SISO stage, enabling iterative (turbo) decoding loops.
- **Punctured rate-3/4 operation** ? supply $L_{c_i}[t]=0$ on punctured positions to decode higher-rate punctured variants on the same trellis.
- **Soft-output equalization chains** ? feeds reliability metrics to downstream LDPC, Reed?Solomon, or CRC-assisted list decoders.
- **Performance/BER analysis** ? the LLR magnitude $|L(u_t)|$ provides a per-bit confidence measure for link-margin and error-floor studies.

---

## 9. Notes & Limitations

- This is a **Max-Log-MAP** approximation, not full log-MAP; it omits the Jacobian correction term $\log\!\big(1 + e^{-|x_1 - x_2|}\big)$, trading a fraction of a dB of coding gain for branchless, transcendental-free arithmetic.
- The trellis assumes a **known start state** ($s_0 = 0$) and a **free, non-terminated end state**.
- The $C_2$ inversion **must** be applied to `lc2` by the caller (or pre-flipped upstream) to match the inverted-$C_2$ trellis.
