# Self-Rewarding Double DQN (SRDDQN) for Crypto Trading
**Research & Implementation Notes**

## 1. Project Overview & Architecture
This project successfully implemented a state-of-the-art Self-Rewarding Double Deep Q-Network (SRDDQN) algorithmic trading agent. Our goal was to solve the classic reinforcement learning (RL) problem in finance: RL agents often overfit to micro-fluctuations (noise) and get killed by transaction fees due to over-trading.

To solve this, we replicated the architecture of recent financial ML papers:
1. **Multi-Timeframe (MTF) Data Pipeline**: We combined 1H, 4H, and 1D technical indicators (RSI, MACD, Bollinger Bands, etc.) into a flattened 27-feature observation space to give the agent spatial context.
2. **TimesNet Pre-training**: We implemented the highly advanced TimesNet model (using Fast Fourier Transforms to extract dominant market periodicities) and trained it via Supervised Learning to predict the perfect-hindsight "Min-Max" macro trend.
3. **Self-Rewarding Mechanism**: We wrapped the base `CryptoEnv` in a custom `SRDRLWrapper`. During RL training, the agent's reward is dynamically updated using the TimesNet's prediction: `r_t = max(r_env, r_timesnet)`. This forces the agent to learn to hold through macro-trends rather than reacting to 1H noise.

## 2. Training Journey & Benchmark Results

We progressed through 5 phases, incrementally upgrading the agent. The final evaluation was run on an unseen validation dataset with realistic 0.1% transaction fees.

| Metric | Phase 2 (PPO Basic) | Phase 3 (PPO + MTF) | **Phase 5 (SRDDQN)** |
| :--- | :--- | :--- | :--- |
| **Cumulative Return** | -96.81% | -23.96% | **+2.91%** |
| **Annualized Return** | -97.33% | -74.38% | **+24.35%** |
| **Sharpe Ratio** | -9.61 | -1.85 | **+0.7167** |
| **Max Drawdown** | 96.84% | 28.81% | **17.71%** |
| **Final Equity** | $319 | $7,604 | **$10,291** |

### Visualization: SRDDQN Entries & Exits
The agent successfully learned to ignore the chop and hold positions during clear macro trends.

![SRDDQN Trading Entries and Exits](logs/plots/srddqn_trading_visualization.png)
*(Green Triangle = Long Entry, Red Triangle = Short Entry, Blue Cross = Exit to Flat)*

---

## 3. Future Roadmap & Refinements

While this initial prototype shows incredible promise and profitability, moving from a research prototype to a production live-trading algorithm requires significant hardening. Here is the recommended roadmap:

### Phase 1: Robust Testing & Walk-Forward Validation
1. **Walk-Forward Validation (WFV)**: Our current train/val split is a simple 80/20 chronological split. Markets change regimes. Implement a rolling Walk-Forward Validation pipeline (e.g., Train on 2018-2020, Test on 2021; then Train 2019-2021, Test 2022).
2. **Hyperparameter Optimization**: Use **Optuna** to optimize both the TimesNet architecture (number of Inception blocks, `top_k` frequencies) and the RL hyperparameters (learning rate, buffer size, gamma, epsilon decay).
3. **Slippage & Market Impact Models**: We currently simulate a flat 0.1% fee. In reality, large market orders suffer from slippage. Implement a volume-based slippage penalty in the environment.

### Phase 2: Feature Engineering & Multi-Asset Scaling
1. **Alternative Data**: Price action is only half the picture. Integrate orderbook imbalance, funding rates, open interest, and liquidations into the observation space.
2. **Multi-Asset Portfolio**: Currently, the agent trades a single asset (BTC/USDT). Extend the environment to handle a portfolio of top N altcoins. TimesNet excels at finding cross-asset correlations if fed a multi-asset matrix.

### Phase 3: Live Inference & Paper Trading
1. **Live Data Ingestion**: Transition the `data_pipeline` from reading historical Parquet files to connecting to a Binance or Bybit WebSocket feed. Maintain a rolling in-memory buffer of the last `N` hours to calculate the MTF features on the fly.
2. **Asynchronous Execution Engine**: The RL agent operates in discrete timesteps (e.g., once an hour). You need a lightweight execution script that wakes up on the hour, queries the WebSocket for the latest 1H close, feeds the MTF features to the ONNX-exported SRDDQN model, and translates the integer action into a REST API order.
3. **Paper Trading Phase**: Run the live execution engine with API keys restricted strictly to "Read-Only" or connected to the exchange's Testnet. Run this for a minimum of 3-4 weeks to prove that the live-executed returns match the backtest simulations (this is crucial to ensure there is zero look-ahead bias in the real-time feature calculation). 

### Phase 4: Production Live Trading
1. **Risk Management Guardrails**: Never trust the neural network blindly. Implement hard-coded safety triggers in the execution engine (e.g., "If portfolio drawdown exceeds 15%, force close all positions and halt trading").
2. **Deployment**: Containerize the execution engine via Docker and deploy it to an AWS EC2 instance located physically close to the exchange's servers (e.g., Tokyo for Binance) to minimize latency.
