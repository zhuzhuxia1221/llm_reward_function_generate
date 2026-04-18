import os
import time
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=API_KEY)

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},  
}

PROMPT = """
You are designing reward functions for a Deep Reinforcement Learning (DRL) environment that schedules computational tasks to edge servers in a threats-based UND variant.
============================
ENVIRONMENT DEFINITIONS (STRICT CONSTRAINTS)
============================
- The reward function must be defined exactly as:
    def get_reward(self, state, action):
        ...

- Parameter unpacking of state must be:
    gram, gcores, gwl_c, gwl_g, gs1_old, gs2_old = state

- The action must be converted to capacity as follows:
    u1 = (action // 5) * 4 + 1
    u2 = (action % 5) * 4 + 1

- The environment has a DataFrame `self.dual_s_base` (NOT self.df). You must always use `self.dual_s_base` in your code.
  The columns (exact names) are:
    ram, cores, workload_cpu, workload_gpu, users_yolo, users_mnet, time_yolo, time_mnet

- You must filter the DataFrame with:
    fetch_state = self.dual_s_base.loc[
        (self.dual_s_base['ram'] == gram) &
        (self.dual_s_base['cores'] == gcores) &
        (self.dual_s_base['workload_cpu'] == gwl_c) &
        (self.dual_s_base['workload_gpu'] == gwl_g) &
        (self.dual_s_base['users_yolo'] == gs1) &
        (self.dual_s_base['users_mnet'] == gs2)
    ]

- If `fetch_state` is empty, return a fixed negative reward (e.g., -20).

- Latency values must be obtained as:
    time1 = fetch_state.sample().iloc[0]['time_yolo']
    time2 = fetch_state.sample().iloc[0]['time_mnet']
    tm = max(time1, time2)

- The latency threshold is stored in `self.latency_threshold` and must be used as:
    if tm <= self.latency_threshold:
        ...
    else:
        ...

============================
OPTIMIZATION GOALS
============================
You must produce **ten candidate reward functions**. 
All ten candidates must have the same optimization goal:

- **Maximize the number of scheduled users under latency constraints.**

The reward functions must be different in structure or formulation,
but they must all follow the same optimization target above.


============================
RULES
============================
- All functions must still penalize configurations with latency above the threshold.
- Each function must visibly differ in weighting and structure, not just small coefficient changes.
- The reward structure must make the optimization focus obvious from the code.
- Output format:
    # Candidate 1: <name>
    def get_reward(self, state, action):
        ...
- Each function must be complete and runnable inside the `MECEnvThreatsUnd` environment without modification.
- You MUST use `self.dual_s_base` instead of `self.df` to avoid missing attribute errors.
- Do NOT include markdown code fences (such as ```python or ```), only return raw Python code.
- The differences should come from different mathematical expressions, weighting, or transformations, but not from changing the optimization goal.

"""

MODEL = "gpt-4o-mini"

print("Sending prompt to OpenAI API...")
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

usage = response.usage
input_tokens = usage.input_tokens
output_tokens = usage.output_tokens
total_tokens = usage.total_tokens

if MODEL in PRICING:
    price = PRICING[MODEL]
    cost_input = (input_tokens / 1_000_000) * price["input"]
    cost_output = (output_tokens / 1_000_000) * price["output"]
    total_cost = cost_input + cost_output
else:
    total_cost = 0.0
    print(f"No pricing info for model {MODEL}, cost=0")

print(f"Token usage: input={input_tokens}, output={output_tokens}, total={total_tokens}")
print(f"API Latency: {latency:.2f} seconds")
print(f"Estimated cost: ${total_cost:.6f}")

output_dir = "reward_candidates"
os.makedirs(output_dir, exist_ok=True)

candidates = content.split("# Candidate")
saved_files = []
counter = 1
for code in candidates:
    if "def get_reward" in code:
        file_path = os.path.join(output_dir, f"AAnew_reward_candidate_threat1_{counter}.py")
        with open(file_path, "w") as f:
            f.write("# Candidate" + code.strip())
        saved_files.append(file_path)
        counter += 1

print(f"Generated {len(saved_files)} candidate reward functions:")
for path in saved_files:
    print(f"   {path}")
print("You can now replace mec_env.py's get_reward with one of these for training.")
