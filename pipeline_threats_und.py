import os
import re
import subprocess
import pandas as pd

CANDIDATES_DIR = "reward_candidates"
ENV_FILE = "mec_env_threats_und.py"
TRAIN_FILE = "train_rl_threats_und.py"
ALLOC_FILE = "allocation_threat_und.py"

def insert_reward_function(env_file, candidate_file):
    with open(env_file, "r") as f:
        env_code = f.read()

    with open(candidate_file, "r") as f:
        reward_code = f.read()

    reward_code = "\n".join([line for line in reward_code.splitlines() if not line.strip().startswith("# Candidate")])
    reward_code_indented = []
    for idx, line in enumerate(reward_code.splitlines()):
        if line.strip():
            if idx == 0:  
                reward_code_indented.append("    " + line)
            else:  
                reward_code_indented.append("        " + line)
        else:
            reward_code_indented.append("")
    reward_code_indented = "\n".join(reward_code_indented)

    env_code_new = re.sub(
        r" def get_reward\(self, state, action\):[\s\S]*?(?=\n    def|\nclass|\Z)",
        reward_code_indented + "\n",
        env_code
    )

    with open(env_file, "w") as f:
        f.write(env_code_new)

def run_pipeline(candidate_idx):
    candidate_file = os.path.join(CANDIDATES_DIR, f"AAnew_reward_candidate_threat_{candidate_idx}.py")
    if not os.path.exists(candidate_file):
        print(f"Candidate {candidate_idx} not found.")
        return

    print(f"\n=== Processing Candidate {candidate_idx} ({os.path.basename(candidate_file)}) ===")

    print("Inserting reward function into environment...")
    insert_reward_function(ENV_FILE, candidate_file)

    model_path = f"saved_models/und_agent_candidate_{candidate_idx}.zip"
    print(f"Training model for candidate {candidate_idx} ...")
    subprocess.run([
        "python3", TRAIN_FILE,
        "--output_model", model_path
    ], check=True)

    print(f"Testing model for candidate {candidate_idx} ...")
    subprocess.run([
        "python3", ALLOC_FILE,
        "--model_path", model_path,
        "--output", f"allocation_results/und_results_candidate_{candidate_idx}.csv"
    ], check=True)

def summarize_results(num_candidates=5, output_file="allocation_results/und_results_summary.csv"):
    all_ilp = []
    all_greedy = []
    rl_results = {}

    for idx in range(1, num_candidates + 1):
        file_path = f"allocation_results/und_results_candidate_{idx}.csv"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, skipping.")
            continue

        df = pd.read_csv(file_path)

        if idx == 1:  
            users = df["user"]

        all_ilp.append(df["ilp_user"])
        all_greedy.append(df["greedy_user"])
        rl_results[f"rl_user_candidate_{idx}"] = df["rl_user"]

    ilp_mean = pd.concat(all_ilp, axis=1).mean(axis=1).round().astype(int)
    greedy_mean = pd.concat(all_greedy, axis=1).mean(axis=1).round().astype(int)

    summary_df = pd.DataFrame({
        "user": users,
        "ilp_user": ilp_mean,
        "greedy_user": greedy_mean,
    })

    for key, values in rl_results.items():
        summary_df[key] = values

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")

def main():
    candidates = [f for f in os.listdir(CANDIDATES_DIR) if f.startswith("AAnew_reward_candidate_threat_")]
    candidates.sort()
    print(f"Found {len(candidates)} candidates.")

    for idx in range(1, len(candidates) + 1):
        run_pipeline(idx)

    summarize_results(num_candidates=len(candidates))

if __name__ == "__main__":
    main()
