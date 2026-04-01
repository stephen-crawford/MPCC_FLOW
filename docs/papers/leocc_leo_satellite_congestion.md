# LeoCC: Making Internet Congestion Control Robust to LEO Satellite Dynamics

**Authors:** Zeqi Lai, Zonglun Li, Qian Wu, Hewu Li, Jihao Li, Xin Xie, et al. (Tsinghua University, Beijing Institute of Technology)
**Venue:** ACM SIGCOMM 2025
**DOI:** 10.1145/3718958.3750491

## Core Problem

Low-Earth-Orbit (LEO) satellite networks (Starlink, OneWeb, Kuiper) introduce unique challenges for congestion control:
1. **Rapidly varying link capacity**: fluctuates between 10-70 Mbps in tens of seconds
2. **Frequent RTT fluctuations**: base RTT changes drastically due to satellite path changes
3. **High packet loss rates**: 0.5-6% random loss from space-ground handovers
4. **Connection reconfiguration**: Starlink replans connections every ~15 seconds, causing 45-120ms outages

These dynamics **break the fundamental assumptions** of existing CCAs:
- **Loss-based** (Cubic, Reno): interpret LEO-dynamics losses as congestion
- **Delay-based** (Vegas, Copa): overreact to LEO-induced RTT changes
- **Model-based** (BBR): max-filter bandwidth and min-RTT estimates become stale
- **Learning-based** (VIVACE, Proteus): mapping from observations to rates changes too fast to learn

## Key Insight

LEO networks have a unique feature: **connection reconfiguration** events that are strongly correlated with drastic network variations. By detecting reconfigurations, LeoCC can distinguish LEO-dynamics-induced changes from actual congestion and build an accurate time-varying bottleneck model.

## LeoCC Core Design

### 1. Reconfiguration-Aware Network Model (Section 4.1)

**Reconfiguration Detection:**
- Maintains a light ICMP probing flow to the Point-of-Presence (PoP)
- Monitors Response Intervals (RIs) between consecutive ICMP responses
- Reconfiguration detected when RI exceeds threshold Delta_outage (~45ms)
- Works for both periodic (~15s intervals) and aperiodic reconfigurations

**Dynamic Network Model:**
- Path modeled as sequence of discrete conditions: {(bBW^R, bRTT^R), (bBW^{R+1}, bRTT^{R+1}), ...}
- At time T: bBW(T) = bBW^R + noise_BW(T), bRTT(T) = bRTT^R + noise_RTT(T)
- After reconfiguration R, discard all pre-R measurements

**Bandwidth Estimation (dual estimator):**
- Aggressive: bBW_a(T) = Max(dRate_t) -- max filter over delivery rates
- Moderate: bBW_m(T) = Kalman(dRate_t) -- Kalman filter
- Only use samples since last reconfiguration (or within window W_BW)

**RTT Estimation (band-based):**
- Maintains bRTT range [bRTT_l, bRTT_h] instead of single min-RTT
- bRTT_h = max(MPF(RTT_t)): upper bound via median-pass filter
- bRTT_l = min(MPF(RTT_t)): lower bound
- Also tracks bRTT_latest via Kalman filter for real-time estimate

### 2. Robust Rate Controller (Section 4.2)

**State Machine:**
- **Startup**: rapidly increase sending rate
- **Dynamic Cruise**: adapt within reconfiguration interval
- **Reconfiguration Adaptation**: rapidly converge to new conditions after reconfig

**Target Rate Calculation:**
```
rate_target = bBW_m   if bRTT_latest in [bRTT_l, bRTT_h + Delta_RTT]  (normal)
            = bBW_a   otherwise  (aggressive when uncongested)
```

**Sending Rate via Burst/Drain Cycles (inspired by BBR):**
- Burst coefficient b_co, drain coefficient d_co
- First/second RTT of cycle: send at b_co * rate_target, d_co * rate_target
- Remaining RTTs: send at rate_target
- If bRTT_latest <= bRTT_h + Delta_RTT: b_co=1.25, d_co=0.75 (aggressive)
- Otherwise: slightly decrease b_co (conservative)

**Reconfiguration Adaptation:**
- On reconfiguration detection: limit inflight to 0.5 * bBW_m * bRTT_l
- Wait for queue drain, collect new RTT samples
- Update bRTT_h, bRTT_l immediately from new samples
- Set bBW_a = bBW_m = rate*_target (pre-reconfig rate)
- Return to Dynamic Cruise

### 3. Bottleneck Shifting (Section 4.3)
- Detects whether bottleneck is in satellite segment or terrestrial segment
- Probes RTT from terminal to PoP vs end-to-end RTT
- RTT_non-sat = RTT_e2e - RTT_term-PoP
- If non-satellite RTT spikes: skip reconfiguration adaptation (bottleneck is not LEO)

## Implementation
- **Linux kernel module** using pluggable TCP APIs
- Single probing flow per satellite terminal (shared across all LeoCC flows)
- Uses eBPF to embed reconfiguration info in TCP ACK Options field
- ~5 Kbps probing overhead (0.13% of average uplink capacity)

## Evaluation Highlights

### Live Starlink (3 terminals: Madrid, New Jersey, Cebu)
- **Pareto-optimal**: LeoCC sits on the upper-left of the throughput-delay Pareto frontier
- vs Cubic: 253% higher throughput (uplink Madrid->Barcelona)
- vs Copa: 494% higher throughput
- vs BBRv1: 85% higher throughput, 44% lower delay
- vs BBRv3: similar throughput, 56% lower delay
- vs VIVACE/Proteus: 56% higher throughput, 38% lower delay

### Trace-Driven Testbed (LeoReplayer, 4.8K traces)
- Consistent Pareto-frontier performance across replayed conditions
- 95.2% link utilization (vs 70.5% Verus, 81.5% VIVACE)

### Fairness
- Self-fairness: Jain index 0.99-1.0 among 4 parallel LeoCC flows
- RTT-fairness: equal throughput despite 20/48/85/145ms RTTs
- TCP coexistence: fair sharing with BBRv1 flows

### Robustness
- Insensitive to AQM strategy (FIFO, CoDel, PIE, HDrop)
- Maintains performance under bad weather and physical obstructions
- Works in conventional wired networks (comparable to BBR)

## Relevance to MPCC Flow

LeoCC addresses a **closely related problem domain** -- congestion control under rapidly varying, unpredictable network conditions. Key connections:

### What to Learn From
- **Reconfiguration-aware model**: the idea of treating network state as a *piecewise-stationary* process with discrete transitions maps well to our MPC formulation -- each reconfiguration interval could be a "planning segment"
- **Dual bandwidth estimator**: aggressive (Max) + moderate (Kalman) estimators parallel our scenario-based constraint approach (optimistic vs conservative)
- **RTT band estimation**: maintaining a range [bRTT_l, bRTT_h] rather than a single value is analogous to our uncertainty ellipses for obstacle prediction
- **State machine with explicit mode transitions**: Startup -> Dynamic Cruise -> Reconfiguration Adaptation maps to MPC operating modes
- **Burst/drain cycles**: the pacing pattern is a form of structured control input, similar to our MPC control sequences

### What MPCC Flow Should Improve On
- LeoCC is **domain-specific** (LEO reconfiguration detection); our MPCC should generalize the "event detection -> model update -> rate adaptation" pattern to any cellular network
- LeoCC's rate controller is still **heuristic** (hand-tuned b_co, d_co, Delta_RTT thresholds); our optimization-based approach should derive these from the cost function
- LeoCC does **not explicitly optimize over a horizon**; it makes single-step rate decisions. Our MPC formulates a multi-step optimization
- LeoCC **lacks formal convergence guarantees**; our MPCC with AIMD compatibility should provide these
- LeoCC is a **kernel module**; our MPCC should also work at user-space (CCP/Nimbus) level for easier deployment

### Comparison Context
- LeoCC references **ABC** (explicit congestion control for wireless) as requiring access point modifications -- exactly the "legacy" approach we must coexist with
- LeoCC references **Verus** as a baseline -- another paper in our analysis set
- The evaluation methodology (real traces + controlled replay) matches our Mahimahi-based approach
