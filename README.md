# DWS-Bench: Dynamic World-State Reasoning Benchmark

A deterministic state-machine benchmark and evaluation framework for evaluating dynamic world-state reasoning in Small Instruction-Tuned Language Models (SLMs) and LLMs from natural-language event narratives.

---

## 1. Core Principles & Methodology

1. **The Simulator is the Source of Truth**: Every natural-language instance is rendered from a verified canonical symbolic state-transition sequence executed on a deterministic world simulator.
2. **Five-Group Capability Taxonomy**: Evaluates 8 distinct trajectory families across 5 capability groups.
3. **Decoupled Factor Measurement (§9)**: The generator independently measures actual factors from the symbolic trace:
   $$\boxed{(E,T,D,V,L)_{\text{actual}} = f(\text{canonical symbolic trace}) \neq (E,T,D,V,L)_{\text{requested}}}$$
   protecting empirical comparisons against generator assumptions, ambiguity, and unintended operations.
4. **Controlled Isolation**: Experimental dimensions (temporal depth $T$, state-level distractor interference $D$, revision complexity $V$, and entity load $E$) are varied independently to establish unconfounded behavioral baselines.
5. **Separation of Evidence and Mechanisms**: Behavioral degradation curves and failure onset points are established experimentally before testing mechanistic or attention-based hypotheses.

---

## 2. Five-Group Capability Taxonomy & 8 Trajectory Families

| Capability Group | Trajectory Family | State-Transition Semantics | Targeted Failure Phenomenon / Hypothesis |
|---|---|---|---|
| **A. Sequential State Tracking** (RQ1, RQ2) | `basic_chain` | Pure temporal depth ($E=1, D=0$) with sequential `Move` updates | Degradation with temporal depth |
| | `revision` | Sequential moves with repeated location revisits ($V \ge 2$) | Interference from previously established state |
| **B. Multi-Entity Interference** (RQ3) | `interleaved_chain` | Target entity updates strictly interleaved with distractor entity moves | Susceptibility to irrelevant state-transition interference |
| **C. Identity Transformation** (RQ5 Pilot) | `split_chain` | Dynamic entity multiplication via `Split` | Difficulty tracking identity branching & child states |
| | `merge_chain` | Container-level entity consolidation via `Merge` | Over-persistence (failing container-level relocation) |
| **D. Global State Operations** (RQ5 Pilot) | `swap_chain` | Simultaneous bilateral container exchange via `Swap` | Relational state exchange / unilateral overwrite error |
| **E. Temporal Edit History** (RQ5 Pilot) | `undo_chain` | Rollback of operations via `Undo` | Historical state reversal / treating undone actions as real |
| | `undo_redo_chain` | 3-way history awareness via `Undo` + `Redo` | History re-application & cycle tracking |

---

## 3. Experimental Factor Formalism

We strictly distinguish between state-level factors, textual factors, and generation bookkeeping:

- **$E$ (Entity Load)**: Total number of entities instantiated across the trajectory lifetime.
- **$T$ (Target-Relevant Depth)**: Count of post-initialization operations that causally alter the target entity's state or location.
- **$D$ (Distractor State Updates)**: Count of post-initialization state-changing operations on non-target entities (semantic/state-level interference).
- **$V$ (Revision Complexity)**: Count of genuine location revisits where the target returns to a previously occupied location after intervening transitions.
- **$L$ (Rendered Token Length)**: Token/word length of the rendered narrative text ($L < 600$).
- **$N$ (Textual Distractors)**: Count of optional pure natural-language distractor sentences containing zero state transitions.
- **$U$ (Bookkeeping Variable)**: Post-initialization operations ($U = T + D$).

---

## 4. Research Questions (RQ1–RQ5)

```text
RQ1: How does temporal depth (T) affect dynamic state tracking under single-entity conditions (E=1, D=0, V=0)?
RQ2: How does state revision (V) affect dynamic state tracking at matched temporal depth (E=1, D=0, V>=2)?
RQ3: How does irrelevant state-transition interference (D) on constant entity load (E=3, T=8) affect tracking?
RQ4: How does entity load (E) affect dynamic world-state reasoning?
RQ5 (Pilot): How do qualitatively different state operations (Split, Merge, Swap, Undo, Redo) affect reasoning at common depth (T=8)?
```

### Experimental Sweeps:

#### RQ1: Temporal Depth Sweep ($E=1, D=0, V=0$)
Clean unconfounded baseline testing how models maintain dynamic state as temporal depth increases:
- **Conditions**: $T \in \{2, 4, 6, 8, 12, 16\}$, 100 instances each = **600 instances**.
- **Run**: `python3 experiments/rq1_depth.py`

#### RQ2: Revision Complexity Sweep ($E=1, D=0, V \ge 2$)
Evaluates the impact of revising previously established states at equivalent temporal depth:
- **Conditions**: $T \in \{4, 8, 12, 16\}$, 100 instances each = **400 instances** (control is RQ1 $V=0$).
- **Run**: `python3 experiments/rq2_revision.py`

#### RQ3: Multi-Entity Interference Sweep ($E=3, T=8, V=0$)
Evaluates resistance to irrelevant state-transition interference while holding entity load and target depth constant:
- **Conditions**: $D \in \{4, 8, 16\}$ (with $D=0$ from RQ1 $T=8$) = **300 new instances**.
- **Run**: `python3 experiments/rq3_distractor.py`

#### RQ5: Structural Operation Pilot ($T=8$)
Preliminary pilot assessing whether the broader operation algebra can be reliably generated and evaluated at a normalized depth of $T=8$:
- **Families**: `split_chain` ($E=2$), `merge_chain` ($E=2$), `swap_chain` ($E=2$), `undo_chain` ($E=1$), `undo_redo_chain` ($E=1$).
- **Instances**: 100 per family = **500 instances**.
- **Run**: `python3 experiments/rq5_pilot.py`

---

## 5. Metric Hierarchy & Failure Analysis

### Metric Hierarchy
1. **Primary Outcome ($A_{\text{final}}$)**: Exact-match accuracy on the final query answer.
2. **Trajectory State Accuracy ($A_{\text{step}}$)**: Accuracy of intermediate state tracking across all steps.
3. **Transition Accuracy ($A_{\text{transition}}$)**: Correct application of individual state transitions.
4. **Failure Dynamics**: Failure onset ($L_f$), degradation slope, and error classification.

### Failure Onset ($L_f$) & Curve Fitting
- **Candidate Models**:
  - Linear: $A(x) = a + bx$
  - Exponential Decay: $A(x) = ae^{-bx} + c$
  - Sigmoid: $A(x) = c + \frac{a-c}{1 + e^{b(x - x_0)}}$
- **Primary Selection**: **Akaike Information Criterion (AIC)** (penalizes parameter count); $R^2$ reported as descriptive fit.
- **Operational Failure Onset**: $L_f = \min \{ x : A(x) < \tau \}$ (evaluated at operational thresholds $\tau \in \{0.60, 0.70, 0.80\}$).
- **Multi-Dimensional Failure Profile**: $M = (L_T, L_D, L_V, L_E)$.

### Formal Checkpoint Error Classification
Let binary checkpoint sequence $C_t = \mathbb{I}(\hat{S}_t = S_t)$ compare gold state $S_t$ and predicted intermediate state $\hat{S}_t$:

```text
111111  ->  NO_ERROR            (Exact match across all steps)
110110  ->  LOCAL_ERROR         (Temporary recovery at j > t, but final answer is wrong)
110000  ->  PROPAGATING_ERROR   (Error at step t propagates to all subsequent steps)
111110  ->  FINAL_ONLY_ERROR    (Intermediate states correct; answer extraction/generation fails)
110011  ->  CANCELLATION_ERROR  (Intermediate error occurs, but final answer is restored)
```

---

## 6. Standardized Evaluation Harness & Core Models

Evaluates 5 core instruction-tuned small language models under identical deterministic decoding:

- **Core Models**:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
  - `meta-llama/Llama-3.2-3B-Instruct`
  - `allenai/OLMo-2-1124-7B-Instruct` (or `allenai/OLMo-2-1B-Instruct`)
- **Standardized Decoding**: `temperature=0.0`, `do_sample=False`, `max_new_tokens=128`, standardized zero-shot prompt template.

---

## 7. Naturalistic Reference & ProPara Scope

ProPara serves as an **external naturalistic reference point** rather than a synthetic benchmark replacement:
- Bounded strictly to sequential tracking (`basic_chain`), interference (`interleaved_chain`), and repeated state change (`revision`).
- Structural operations (`Split`, `Merge`, `Swap`, `Undo`, `Redo`) belong exclusively to the controlled synthetic state machine and are not forced into naturalistic corpora.
- Preceded by a 10-paragraph mapping audit to assess expressiveness on natural procedural text.

---

## 8. Verification & Test Suites

Run the complete master test suite:
```bash
python3 test/run_all.py
```

Run dry-run reachability probes:
```bash
python3 experiments/rq1_depth.py --dry-run
python3 experiments/rq2_revision.py --dry-run
python3 experiments/rq3_distractor.py --dry-run
python3 experiments/rq5_pilot.py --dry-run
```
