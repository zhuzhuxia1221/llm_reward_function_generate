# Candidate3: StabilityOptimization
def get_reward(self, state, action):
    u1 = action // 5 + 1
    u2 = (action + 1) - (u1 - 1) * 5
    gram, gcores, gwl_c, gwl_g, gs1_old, gs2_old = state

    gs1 = u1 * 100
    gs2 = u2 * 100

    fetch_state = self.dual_s_base.loc[
        (self.dual_s_base['ram'] == gram) &
        (self.dual_s_base['cores'] == gcores) &
        (self.dual_s_base['workload_cpu'] == gwl_c) &
        (self.dual_s_base['workload_gpu'] == gwl_g) &
        (self.dual_s_base['users_yolo'] == gs1) &
        (self.dual_s_base['users_mnet'] == gs2)
    ]

    if fetch_state.empty:
        return -20

    time1 = fetch_state.sample().iloc[0]['time_yolo']
    time2 = fetch_state.sample().iloc[0]['time_mnet']
    tm = max(time1, time2)

    if tm <= self.latency_threshold:
        return -3 * (abs(gs1 - gs1_old) + abs(gs2 - gs2_old)) + 2 * (u1 + u2)
    else:
        return -3 * (tm - self.latency_threshold)
```