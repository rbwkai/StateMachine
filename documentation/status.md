# DWS-Bench Project Status

**Current Status — August 2026**

The core DWS-Bench generator architecture is implemented and internally verified.

---

### 1. Implemented Components

- **Deterministic World Simulator**: Discrete world-state transitions, container/object mappings, and history rollback/re-application (`world/`).
- **Trajectory Family Constructors**: 8 trajectory constructors implementing the Five-Group Capability Taxonomy (`generator/trajectories.py`).
- **Family-Specific Structural Validation**: Strict validation gates enforcing family invariant properties (`generator/trajectory_validation.py`).
- **QuerySpec-Based Qualification**: Query filtering satisfying formal structural constraints rather than heuristic interestingness (`generator/probes.py`, `analysis/query_analysis.py`).
- **Decoupled Factor Measurement (§9)**: Independent measurement of $(E, T, D, V, L_{\text{word}}, N)_{\text{actual}}$ from canonical trace replay and narrative text (`generator/metadata.py`).
- **Step-Wise Canonical Gold Trajectories**: Full intermediate state ground truth for trajectory tracking evaluation (`trajectory.py`, `generator/probes.py`).
- **Causal Counterfactual Probes**: Single-operation deletion intervention with causal sensitivity labelling (`generator/probes.py`).
- **Deterministic Natural-Language Rendering**: Linguistic surface realization without altering simulator state (`render/`).
- **Experimental Generation Scripts**: Dedicated generator pipelines for RQ1 (Depth), RQ2 (Revision), RQ3 (Interference), and RQ5 (Structural Pilot) (`experiments/`).
- **Failure Analysis Infrastructure**: Linear, exponential, and sigmoid curve fitting with AIC model selection, failure onset ($L_f$), and 5-way first-error classification (`analysis/`).
- **Standardized Evaluation Harness**: Deterministic decoding configurations for 5 core instruction-tuned SLMs and answer extraction pipeline (`eval/`).

---

### 2. Verified Properties & Automated Test Suites

Verified via master test runner (`python3 test/run_all.py`):

| Test Suite | Scope | Status |
|---|---|---|
| `test/smoke_test.py` | State model transitions, replay engine, and query pipelines | **PASS** |
| `test/smoke_test_trajectories.py` | Structural validation & determinism across all 8 trajectory families | **PASS** |
| `test/test_invariants.py` | Mathematical invariant consistency over 200 randomized instances | **PASS** |
| `test/test_measured_factors.py` | $(E, T, D, V, L_{\text{word}}, N)_{\text{actual}}$ factor measurement & mismatch detection | **PASS** |
| `test/test_analysis_and_eval.py` | Curve fitting (AIC), failure onset, first-error classification, and eval harness | **PASS** |

All experimental condition reachability probes (`--dry-run`) confirm 10/10 successes for RQ1, RQ2, RQ3, and RQ5 pilot conditions at $T=8$.

---

### 3. What Has Not Yet Been Tested

The current verification establishes **implementation correctness and internal consistency**. It does not yet establish:
- Empirical benchmark-level factor independence across large generation runs.
- Model-level performance degradation curves on physical small language models.
- Empirical transfer to naturalistic procedural narratives.

---

### 4. Remaining Research Milestones

```text
                 DWS-Bench
                     │
       ┌─────────────┴─────────────┐
       │                           │
Controlled Benchmark        Naturalistic Reference
       │                           │
       ↓                           ↓
Phase 1: Generator Validation    Phase 3: ProPara Audit
(1,000+ instances, factor        (10-paragraph mapping audit,
 correlations & distributions)   transfer check)
       │                           │
       ↓                           │
Phase 2: SLM Evaluation ───────────┘
(5 instruction-tuned models,
 accuracy, failure onset, error
 dynamics & curve fitting)
       │
       ↓
Mechanistic Interpretability & Thesis Conclusions
```

#### Phase 1 — Generator Validation
- Generate 1,000+ instances across RQ1 ($T \in \{2..16\}$), RQ2 ($V \ge 2$), RQ3 ($D \in \{4..16\}$), and RQ5 pilot ($T=8$).
- Measure acceptance rates and generation performance.
- Inspect realized factor distributions and quantify residual correlations among $E, T, D, V, L_{\text{word}}$, and $N$.
- Verify structural diversity across seeds and freeze benchmark datasets.

#### Phase 2 — SLM Evaluation
- Run standardized zero-shot evaluation across the 5 core instruction-tuned models:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
  - `meta-llama/Llama-3.2-3B-Instruct`
  - `allenai/OLMo-2-1124-7B-Instruct`
- Measure final-answer accuracy ($A_{\text{final}}$) and trajectory step accuracy ($A_{\text{step}}$).
- Estimate operational failure onset ($L_f$) and classify error dynamics (Local, Propagating, Cancellation).
- Fit difficulty curves and select models via AIC.

#### Phase 3 — Naturalistic Reference
- Conduct the 10-paragraph ProPara mapping audit.
- Evaluate whether controlled synthetic findings transfer to natural process narratives without conflating synthetic extensions into naturalistic corpora.
