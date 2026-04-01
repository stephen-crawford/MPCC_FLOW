# Verus: Adaptive Congestion Control for Unpredictable Cellular Networks

**Authors:** Yasir Zaki, Thomas Potsch, Jay Chen, Lakshminarayanan Subramanian, Carmelita Gorg
**Venue:** ACM SIGCOMM 2015
**DOI:** 10.1145/2785956.2787498

## Core Problem

Legacy congestion control protocols (TCP Cubic, Reno, Vegas) perform poorly over cellular networks due to:
1. **Highly variable capacities** at short timescales (ms to seconds)
2. **Self-inflicted packet delays** from bufferbloat
3. **Stochastic packet losses** unrelated to congestion
4. **Burst scheduling** at base stations creating unpredictable inter-arrival patterns
5. **Competing traffic** causing cross-flow dependencies even with per-user queues

## Key Insight

Instead of trying to *predict* cellular channel dynamics (which Verus finds intractable), Verus continuously **explores** the delay-vs-window relationship by learning a **delay profile** -- a mapping from sending window size to expected delay. The protocol adapts the window using small epsilon-steps every epoch, staying in constant exploration mode.

## The Verus Protocol

### 1. Delay Estimator
- Tracks RTT for every packet via ACK timestamps
- Computes maximum delay per epoch using EWMA:
  - D_max,i = alpha * D_max,i-1 + (1 - alpha) * max(D_vec_i)
- Delta_D_i = D_max,i - D_max,i-1 (increase/decrease signal)

### 2. Delay Profile
- Core data structure: graph of (sending_window W, packet_delay D) pairs
- Built during slow start, updated continuously via EWMA
- Cubic spline interpolation over (W, D) tuples
- Updated at configurable intervals (~1s) to track channel changes
- As channel degrades, profile steepens; as it improves, profile flattens

### 3. Window Estimator
- Uses Delta_D_i to decide direction:
  - If D_max,i / D_min > R: reduce aggressively (D_est -= delta_2)
  - Elif Delta_D_i > 0: reduce moderately (D_est = max[D_min, D_est - delta_1])
  - Else: increase (D_est += delta_2)
- Looks up D_est on the delay profile to get target window W_{i+1}
- R parameter trades off throughput vs delay (R=2: low delay; R=6: high throughput)

### 4. Epoch-Based Sending
- Sending window divided into small epsilon-ms epochs (epsilon = 5ms)
- Packets to send per epoch: S_{i+1} = max[0, (W_{i+1} + (2-n)/(n-1) * W_i)]
  - where n = ceil(RTT / epsilon)
- This smoothly transitions from current window to target

### 5. Loss Handler
- On loss/timeout: W_{i+1} = M * W_loss (multiplicative decrease)
- Enters loss recovery phase (delay profile frozen)
- Recovery: additive increase 1/W_{i+1} per ACK (like TCP)

## Key Parameters
| Parameter | Value | Effect |
|-----------|-------|--------|
| epsilon | 5 ms | Epoch length; smaller = faster adaptation |
| delta_1 | 1 ms | Conservative decrement |
| delta_2 | 2 ms | Aggressive decrement/increment |
| R | 2-6 | Max tolerable D_max/D_min ratio |
| alpha | 0 < alpha <= 1 | EWMA smoothing for D_max |
| M | multiplicative | Decrease factor on loss |

## Evaluation Highlights

### Real-World (Etisalat 3G/LTE, UAE)
- **vs TCP Cubic**: >10x delay reduction on 3G and LTE, comparable throughput
- **vs Sprout**: Higher throughput in rapidly fluctuating channels, similar delay on LTE
- **vs TCP Vegas**: Similar delay, higher throughput

### Trace-Driven Simulation (OPNET)
- Tested with 2-20 competing flows
- Jain's fairness index: 87-95% (comparable to TCP NewReno, better than Cubic at high contention)
- Quickly adapts to new arriving flows (fair share within seconds)

### RTT Fairness
- With 3 flows at RTTs 20/50/100ms: all achieve similar throughput
- Close to Max-Min fairness (RTT-independent)

### TCP Coexistence
- 3 Verus + 3 TCP Cubic sharing 60 Mbps: fair bandwidth sharing
- Verus does not starve TCP flows

## Key Design Properties

1. **Delay-based with exploration**: combines delay signals with continuous probing
2. **Multiplicative decrease on loss**: retains TCP-compatible loss response
3. **No channel prediction**: avoids the pitfall of inaccurate prediction models
4. **Adaptive delay profile**: learns and re-learns the window-delay relationship
5. **End-to-end**: no router modifications needed
6. **UDP-based implementation**: uses librt for real-time scheduling

## Relevance to MPCC Flow

Verus is a **key comparison target** and design inspiration for our MPCC congestion controller.

### What to Learn From
- **Delay profile as learned constraint**: the (W, D) curve is analogous to a learned constraint boundary in our MPC formulation; we could represent this as a contouring path
- **Sliding window with AIMD**: Verus uses additive increase (delta steps) and multiplicative decrease on loss -- exactly the AIMD paradigm our algorithm must incorporate
- **Epoch-based sending**: the epsilon-epoch structure maps to our MPC timestep
- **R parameter for throughput-delay tradeoff**: analogous to tuning weights in our contouring objective (contour_weight vs lag_weight)
- **Fairness properties**: Verus achieves good fairness through delay-based control; our MPCC must match or exceed this

### What MPCC Flow Should Improve On
- Verus's delay profile is **reactive** (updated every ~1s with spline re-interpolation); our MPC should use a *predictive* model over the horizon
- Verus has **no formal convergence proof**; the authors note that "delay-based protocols are harder to model analytically" -- our MPCC formulation should fill this gap
- Verus's window estimation is **heuristic** (delta_1, delta_2, R are hand-tuned); our optimization-based approach should derive window adjustments from the cost function
- Verus does **not model bandwidth evolution dynamics**; it assumes the delay profile captures the relationship implicitly
- Verus's delay profile can become stale during rapid changes (1s update interval); our MPC with shorter horizons should react faster

### MPCC Design Mapping
| Verus Concept | MPCC Flow Analog |
|---------------|-----------------|
| Delay profile | Reference path (contouring path in throughput-delay space) |
| Window W | Control input (sending rate) |
| Delta_D signal | State feedback (measured delay change) |
| Epoch epsilon | MPC timestep |
| R parameter | Contouring weight ratio |
| Loss handler | Multiplicative decrease constraint |
| Slow start | MPC warmstart / initialization |
