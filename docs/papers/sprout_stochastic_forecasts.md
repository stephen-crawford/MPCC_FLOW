# Sprout: Stochastic Forecasts Achieve High Throughput and Low Delay over Cellular Networks

**Authors:** Keith Winstein, Anirudh Sivaraman, Hari Balakrishnan (MIT CSAIL)
**Venue:** NSDI 2013 (10th USENIX Symposium on Networked Systems Design and Implementation)

## Core Problem

Cellular wireless networks have rapidly varying link rates and deep packet queues in network gateways. Existing transport protocols (TCP Cubic, Vegas, LEDBAT, Compound TCP) and interactive applications (Skype, Hangout, Facetime) react too slowly to rate changes, causing either:
- **Over-buffering**: multi-second standing queues when sending too aggressively
- **Under-utilization**: wasted capacity when being too conservative

## Key Insight

Instead of reactive congestion control (loss/delay signals), Sprout uses **receiver-side stochastic forecasting** of future link capacity. The receiver observes packet arrival times, infers the underlying link rate via Bayesian inference, and sends a *cautious forecast* (5th percentile) of future deliveries to the sender.

## The Sprout Algorithm

### 1. Network Model (Doubly-Stochastic Poisson Process)
- Link modeled as a Poisson process with rate lambda
- lambda itself varies via Brownian motion with noise power sigma (sigma = 200 MTU-pps/sqrt(sec))
- Outage state (lambda = 0) has escape rate lambda_c = 1
- Discretized: 256 possible lambda values (0 to 1000 MTU-pps = 11 Mbps)

### 2. Bayesian Inference (every 20ms tick)
- **Evolve**: apply Brownian motion to probability distribution over lambda
- **Observe**: multiply each P(lambda=x) by Poisson likelihood of observed byte count k:
  - F(x) = P_old(lambda=x) * (x*tau)^k / k! * exp(-x*tau)
- **Normalize**: P_new(lambda=x) = F(x) / sum(F)

### 3. Packet Delivery Forecast
- For each of next 8 ticks (160ms), evolve distribution forward without observation
- Compute 5th percentile of cumulative deliveries at each tick
- This is the *cautious* forecast: 95% probability that at least this many bytes will be delivered

### 4. Control Protocol
- Receiver piggybacks forecast onto ACK packets
- Sender computes **window size** = forecast of bytes to be drained in 5 ticks (100ms) minus current queue occupancy estimate
- Combines elements of pacing (forecast-driven) and window-based flow control
- Sender tracks queue occupancy: increments by bytes sent, decrements by forecast drain per tick

### 5. Simplification: Sprout-EWMA
- Replaces Bayesian inference with exponentially-weighted moving average of throughput
- Predicts constant rate for next 8 ticks
- Higher throughput but also higher delay than Sprout (no cautious forecasting)

## Evaluation Highlights (Cellsim trace replay)
- **Networks**: Verizon LTE, Verizon 1xEV-DO (3G), AT&T LTE, T-Mobile 3G UMTS
- **Compared against**: Skype, Hangout, Facetime, TCP Cubic, Vegas, LEDBAT, Compound TCP, Cubic-CoDel

| vs. | Throughput gain | Delay reduction |
|-----|---------------|-----------------|
| Skype | 2.2x | 7.9x (2.52s -> 0.32s) |
| Hangout | 4.4x | 7.2x (2.28s -> 0.32s) |
| Facetime | 1.9x | 8.7x (2.75s -> 0.32s) |
| Cubic | 0.91x | 79x (25s -> 0.32s) |
| Cubic-CoDel | 0.70x | 1.6x (0.50s -> 0.32s) |

- Sprout-EWMA outperforms all methods on throughput (including Cubic)
- Sprout achieves lowest delay across all 8 cellular links
- SproutTunnel isolates interactive from bulk flows: +528% Skype throughput, -97% Skype delay when coexisting with Cubic

## Key Design Properties

1. **End-to-end**: no router/gateway modifications needed
2. **Proactive, not reactive**: forecasts future capacity rather than reacting to loss/delay
3. **Probabilistic safety bound**: 95% confidence that queuing delay stays < 100ms
4. **Self-clocking via forecast**: window evolves based on predicted channel state, not ACK clocking
5. **Handles outages**: Bayesian model has sticky outage state; recovers gracefully

## Limitations
- Not designed for cross-traffic contention (assumes isolated per-user queue at base station)
- Throughput lower than Cubic (trades throughput for delay control)
- Sprout-EWMA beats Sprout on throughput but can't match it on delay
- Only tested on 2012-era cellular traces

## Relevance to MPCC Flow

Sprout is a **primary comparison target** for our MPCC congestion controller. Key connections:

### What to Learn From
- **Stochastic forecasting**: Sprout's Bayesian inference over a varying-rate Poisson process is conceptually similar to our MPC's use of obstacle prediction modes; we should forecast *bandwidth evolution* similarly
- **Probabilistic safety bound**: Sprout guarantees 95% queue-drain probability, analogous to our scenario-based constraint satisfaction
- **Window-based control with forecast**: the evolving window driven by forecasts maps directly to our sliding-window feedback mechanism

### What MPCC Flow Should Improve On
- Sprout's model is **fixed** (sigma, lambda_c chosen once); our MPC should adapt its dynamics model online
- Sprout uses a **simple window** (single scalar); our MPCC operates over a *horizon* of future send rates
- Sprout does **not handle multi-flow fairness**; our design must provably converge with competing flows
- Sprout's forecast is **1D** (bytes); our contouring framework can optimize over multiple objectives simultaneously (throughput, delay, fairness)
- Sprout has **no explicit convergence proof**; our AIMD-compatible MPCC must prove convergence

### Integration Opportunity
- Sprout's Cellsim trace-replay testbed is the predecessor of **Mahimahi** (already in our repo)
- Our network_emulation/ module with Mahimahi traces provides the same experimental methodology
