import os
import time
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=API_KEY)

PROMPT = """
In the previous stage, we generated ten candidate reward functions for the UND variant of the DRL environment. 
After testing all of them, we selected the three best-performing reward functions based on scheduling performance.

Now, your task is to **reflect on these three top-performing reward functions** 
and design **ten new candidate reward functions** for the next round of training and evaluation.

============================
ENVIRONMENT DEFINITIONS (STRICT CONSTRAINTS)
============================
- The reward function must be defined exactly as:
    def get_reward(self, state, action):
        ...

- Parameter unpacking of state must be:
    gram, gcores, gwl_c, gwl_g, gs1_old, gs2_old = state

- The action must be converted to user allocation as follows:
    u1 = (action // 5) * 4 + 1
    u2 = (action % 5) * 4 + 1

- The environment has a DataFrame `self.dual_s_base` (NOT self.df). Always use `self.dual_s_base`.
  The columns (exact names) are:
    ram, cores, workload_cpu, workload_gpu, users_yolo, users_mnet, time_yolo, time_mnet

- You must filter the DataFrame with:
    fetch_state = self.dual_s_base.loc[
        (self.dual_s_base['ram'] == gram) &
        (self.dual_s_base['cores'] == gcores) &
        (self.dual_s_base['workload_cpu'] == gwl_c) &
        (self.dual_s_base['workload_gpu'] == gwl_g) &
        (self.dual_s_base['users_yolo'] == u1) &
        (self.dual_s_base['users_mnet'] == u2)
    ]

- If `fetch_state` is empty, return a fixed negative reward (e.g., -20).

- Latency values must be obtained as:
    time1 = fetch_state.sample().iloc[0]['time_yolo']
    time2 = fetch_state.sample().iloc[0]['time_mnet']
    tm = max(time1, time2)

- The latency threshold is stored in `self.latency_threshold`:
    if tm <= self.latency_threshold:
        ...
    else:
        ...

============================
REFERENCE: Top-3 Best Performing Reward Functions
============================
# Candidate1: Reward Based on User Count
def get_reward(self, state, action):
    gram, gcores, gwl_c, gwl_g, gs1_old, gs2_old = state
    u1 = (action // 5) * 4 + 1
    u2 = (action % 5) * 4 + 1
    
    fetch_state = self.dual_s_base.loc[
        (self.dual_s_base['ram'] == gram) &
        (self.dual_s_base['cores'] == gcores) &
        (self.dual_s_base['workload_cpu'] == gwl_c) &
        (self.dual_s_base['workload_gpu'] == gwl_g) &
        (self.dual_s_base['users_yolo'] == u1) &
        (self.dual_s_base['users_mnet'] == u2)
    ]

    if fetch_state.empty:
        return -20

    time1 = fetch_state.sample().iloc[0]['time_yolo']
    time2 = fetch_state.sample().iloc[0]['time_mnet']
    tm = max(time1, time2)

    if tm <= self.latency_threshold:
        return u1 + u2  # Reward based on the number of scheduled users
    else:
        return -10  # Penalty for exceeding latency threshold

# Candidate2: Reward Based on Inverse Latency
def get_reward(self, state, action):
    gram, gcores, gwl_c, gwl_g, gs1_old, gs2_old = state
    u1 = (action // 5) * 4 + 1
    u2 = (action % 5) * 4 + 1
    
    fetch_state = self.dual_s_base.loc[
        (self.dual_s_base['ram'] == gram) &
        (self.dual_s_base['cores'] == gcores) &
        (self.dual_s_base['workload_cpu'] == gwl_c) &
        (self.dual_s_base['workload_gpu'] == gwl_g) &
        (self.dual_s_base['users_yolo'] == u1) &
        (self.dual_s_base['users_mnet'] == u2)
    ]

    if fetch_state.empty:
        return -20

    time1 = fetch_state.sample().iloc[0]['time_yolo']
    time2 = fetch_state.sample().iloc[0]['time_mnet']
    tm = max(time1, time2)

    if tm <= self.latency_threshold:
        return (u1 + u2) / (tm + 1)  # Reward inversely proportional to latency
    else:
        return -20  # Strong penalty for exceeding latency threshold
        
# Candidate6: Linear Reward with Direct Latency Penalty
def get_reward(self, state, action):
    gram, gcores, gwl_c, gwl_g, gs1_old, gs2_old = state
    u1 = (action // 5) * 4 + 1
    u2 = (action % 5) * 4 + 1
    
    fetch_state = self.dual_s_base.loc[
        (self.dual_s_base['ram'] == gram) &
        (self.dual_s_base['cores'] == gcores) &
        (self.dual_s_base['workload_cpu'] == gwl_c) &
        (self.dual_s_base['workload_gpu'] == gwl_g) &
        (self.dual_s_base['users_yolo'] == u1) &
        (self.dual_s_base['users_mnet'] == u2)
    ]

    if fetch_state.empty:
        return -20

    time1 = fetch_state.sample().iloc[0]['time_yolo']
    time2 = fetch_state.sample().iloc[0]['time_mnet']
    tm = max(time1, time2)

    if tm <= self.latency_threshold:
        return u1 + u2 - (tm / 5)  # Linear reward with direct latency penalty
    else:
        return -20  # Fixed penalty for exceeding latency threshold      
============================
TASK
============================
Reflect on the above 3 reward functions:
- Identify their common strengths.
- Identify weaknesses and potential improvements.
- Propose **diverse improvements** (not small coefficient tweaks).
- Then design **ten new candidate reward functions** that follow the UND environment rules:
  - Must maximize scheduled users (primary goal).
  - Must penalize latency above threshold.
  - Must show diversity: each candidate should differ clearly in reward shaping, structure, or weighting.
  - Must be directly runnable in MECEnvThreatsUnd without modification.
  
============================
RULES
============================
- The reward structure must make the optimization focus obvious from the code.
- Output format:
    # Candidate 1: <name>
    def get_reward(self, state, action):
        ...
- Each function must be complete and runnable inside the `MECEnvThreatsUnd` environment without modification.
- You MUST use `self.dual_s_base` instead of `self.df` to avoid missing attribute errors.
- Do NOT include markdown code fences (such as ```python or ```), only return raw Python code.

"""

MODEL = "gpt-4o-mini"
print("Sending reflection prompt to OpenAI...")
start_time = time.time()

response = client.responses.create(
    model=MODEL,
    input=PROMPT,
    temperature=0.7,
)
end_time = time.time()
latency = end_time - start_time

content = response.output_text
print("\n====================== LLM RAW OUTPUT START ======================\n")
print(content)
print("\n====================== LLM RAW OUTPUT END ========================\n")

output_dir = "reward_candidates_refined"
os.makedirs(output_dir, exist_ok=True)

candidates = content.split("# Candidate")
saved_files = []
counter = 1
for code in candidates:
    if "def get_reward" in code:
        file_path = os.path.join(output_dir, f"AAnew_reward_candidate_refined1_{counter}.py")
        with open(file_path, "w") as f:
            f.write("# Candidate" + code.strip())
        saved_files.append(file_path)
        counter += 1

print(f"Generated {len(saved_files)} refined reward functions:")
for path in saved_files:
    print(f"   {path}")

usage = response.usage
if usage:
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    total_tokens = usage.total_tokens

    INPUT_COST = 0.15 / 1_000_000   
    OUTPUT_COST = 0.60 / 1_000_000  

    cost_estimate = input_tokens * INPUT_COST + output_tokens * OUTPUT_COST

    print(f"\n Token usage: input={input_tokens}, output={output_tokens}, total={total_tokens}")
    print(f"Estimated cost: ${cost_estimate:.6f} USD")
    print(f"API Latency: {latency:.2f} seconds")
else:
    print("\n Token usage details not available.")