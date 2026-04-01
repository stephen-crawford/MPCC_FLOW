# TURBO: Utility-Aware Bandwidth Allocation for Cloud-Augmented Autonomous Control

**Authors:** Peter Schafhalter, Alexander Krentsel, Hongbo Wei, Joseph E. Gonzalez, Sylvia Ratnasamy, Scott Shenker, Ion Stoica (UC Berkeley)
**Venue:** NINeS 2026 (1st New Ideas in Networked Systems)
**DOI:** 10.4230/OASIcs.NINeS.2026.18

## Core Problem

Autonomous vehicle (AV) ML pipelines require increasingly large models that exceed on-vehicle compute capacity. Cloud offloading is attractive (H100 GPUs are 10x faster than DRIVE Orin), but cellular networks are bandwidth-limited and highly variable, making naive offloading infeasible within strict latency SLOs (150ms for perception, 250ms for motion prediction).

## Key Insight

AV control pipelines are *reconfigurable* DAGs of ML services. Each service has a *family* of models with different accuracy/runtime/data-size tradeoffs. The value of bandwidth differs across services. Co-designing bandwidth allocation with model selection unlocks up to 6x more accuracy improvement than allocating bandwidth without considering which models to run.

## Method: TURBO

### 1. Model-Level Utility (Eq. 1-5)
- Each model m has accuracy A_m and a step utility function:
  - U_m(b) = A_m if total runtime T(b, t_RTT) <= t_SLO, else 0
  - T_remote(b, t_RTT) = t_exec + t_RTT + S_input/b
  - Critical bandwidth: b_c = S_input / (t_SLO - t_exec - t_RTT)
- On-vehicle models have constant utility (no network dependency).

### 2. Service-Level Utility (Eq. 6)
- U_s(b) = max over all models of U_m(b): a staircase function
- Each step = a threshold bandwidth enabling a more accurate cloud model
- On-vehicle model provides a floor guarantee (always available as fallback)

### 3. Application-Level Utility (Eq. 7)
- U_app(b) = max allocation of b across services S, with optional per-service reweighting f_s

### 4. Runtime Bandwidth Allocation (Eq. 8-12)
- Integer Linear Program (ILP) solved at runtime:
  - Binary decision variables x_{s,m} (select model m for service s)
  - Maximize: sum of x_{s,m} * a_{s,m} (total accuracy)
  - Subject to: bandwidth constraint (sum <= B), one model per service
- Solved using PuLP/CBC; runs every 500ms based on CWND and RTT measurements

### 5. Transport
- Uses QUIC (s2n-quic) for concurrent streams without head-of-line blocking
- Monitors CWND and RTT as inputs to the bandwidth allocator

## Evaluation Highlights
- **Simulation** (Waymo Open Dataset v2.0.0, 1150 scenes):
  - +15.6 %pt accuracy over on-vehicle-only baseline
  - +12.7 %pt over naive single-cloud-model offloading
  - Benefits appear at as little as 150 Mbps with 20ms RTT
- **Real-world test drive** (2hr, Ford Explorer, T-Mobile 5G, H100 on GCP):
  - 88% of cloud requests succeeded; +4.1 %pt average accuracy during good connectivity
  - Graceful fallback to on-vehicle models during poor connectivity
- **Dynamic utility** (N-windowed policies): +1.05 %pt over global static policy
- ILP solver runs in <1ms; system overhead is 2.3ms serialization

## Relevance to MPCC Flow

TURBO demonstrates that **optimization-based bandwidth allocation** with utility-aware cost functions significantly outperforms reactive schemes on cellular networks. Key parallels:
- **Utility curves as cost functions**: analogous to MPC stage costs; TURBO's step-function utility per service maps to our contouring/goal objectives
- **ILP over a receding window**: TURBO re-solves every 500ms with current network state, similar to our MPC receding-horizon approach
- **QUIC CWND/RTT as state feedback**: the measured network state feeds the optimizer, just as our sliding-window feedback feeds the MPCC controller
- **Graceful degradation**: on-vehicle fallback guarantees safety, analogous to our constraint-satisfaction guarantees
- **Model family selection**: the idea of choosing different "operating points" based on available capacity parallels AIMD window sizing

### Differences / Gaps MPCC Flow Should Address
- TURBO uses a **static ILP** (no dynamics model of the network); our MPCC approach should incorporate a *dynamics model* of bandwidth evolution
- TURBO's utility is a **step function** (binary: meet SLO or not); our contouring cost can capture *continuous* utility degradation
- TURBO does not prove convergence or fairness analytically; our MPCC formulation should provide **provable convergence** guarantees
- TURBO is single-user; our design must handle **multi-access-point fairness** and coexistence with legacy CCAs
