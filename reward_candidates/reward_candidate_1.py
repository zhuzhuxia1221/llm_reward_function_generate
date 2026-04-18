# Candidate1: Latency-First Optimization
def get_reward(self, state, action):
    u1 = action // 5 + 1
    u2 = (action + 1) - (u1 - 1) * 5
    gram, gcores, gwl_c, gwl_g, gs1_old, gs2_old = state

    gs1 = u1 * 100
    gs2 = u2 * 100

    fetch_state = self.df.loc[
        (self.df['ram'] == gram) &
        (self.df['cores'] == gcores) &
        (self.df['workload_cpu'] == gwl_c) &
        (self.df['workload_gpu'] == gwl_g) &
        (self.df['users_yolo'] == gs1) &
        (self.df['users_mnet'] == gs2)
    ]

    if fetch_state.empty:
        return -50  # Stronger penalty for missing data

    time1 = fetch_state.sample().iloc[0]['time_yolo']
    time2 = fetch_state.sample().iloc[0]['time_mnet']
    tm = max(time1, time2)

    # Strong emphasis on reducing latency
    latency_margin = max(0, tm - self.latency_threshold)  # Positive margin if above threshold

    if tm <= self.latency_threshold:
        return 0.1 * (gs1 - state[4]) + 0.1 * (gs2 - state[5]) - latency_margin
    else:
        return -10 * latency_margin - u1 - u2  # Heavier penalty for exceeding latency
```

```python