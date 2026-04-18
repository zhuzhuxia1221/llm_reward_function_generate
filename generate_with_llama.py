import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/enes/Llama-3.1-8b"   
OUTPUT_DIR = "reward_candidates_llama"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto"
)

end_token_str = "<END>"
end_token_id = tokenizer.eos_token_id


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
- You must generate exactly ten (10) reward functions, no more and no less.
- Once you have generated all 10 new candidate reward functions, output nothing else. 
- End your output with the special token <END>.


"""

print("Sending prompt to LLaMA...")
start_time = time.time()

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=3500,  
        do_sample=True,
        temperature=0.7,
        eos_token_id=end_token_id
    )

end_time = time.time()
latency = end_time - start_time

gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

gen_text = gen_text[len(tokenizer.decode(inputs["input_ids"][0])):]

if end_token_str in gen_text:
    gen_text = gen_text.split(end_token_str)[0]

print("\n====================== LLaMA RAW OUTPUT START ======================\n")
print(gen_text)
print("\n====================== LLaMA RAW OUTPUT END ========================\n")

candidates = gen_text.split("# Candidate")
saved_files = []
counter = 1
for code in candidates:
    if "def get_reward" in code:
        file_path = os.path.join(OUTPUT_DIR, f"AAnew_reward_candidate_llama_{counter}.py")
        with open(file_path, "w") as f:
            f.write("# Candidate" + code.strip())
        saved_files.append(file_path)
        counter += 1

print(f"Generated {len(saved_files)} reward functions:")
for path in saved_files:
    print(f"   {path}")

input_tokens = inputs["input_ids"].shape[1]
output_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
total_tokens = input_tokens + output_tokens

print(f"\n Token usage: input={input_tokens}, output={output_tokens}, total={total_tokens}")
print(f"⏱Latency: {latency:.2f} seconds")