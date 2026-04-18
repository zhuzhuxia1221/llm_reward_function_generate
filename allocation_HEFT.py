import argparse
import numpy as np
import pandas as pd
import geopandas
import random
import matplotlib.pyplot as plt
import timeit
import os
from mip import *
from stable_baselines3 import DQN

def add_data(series):
    random_speed = random.uniform(0, 1)
    if random_speed <= 0.3:
        series['Speed'] = 0
    elif random_speed <= 0.6:
        series['Speed'] = random.uniform(1, 2)
    else:
        series['Speed'] = random.uniform(10, 20)
    series['Direction'] = random.randrange(0, 360, 10)
    qospref = [random.randint(1, 3), random.randint(1, 3)]
    qospref.sort()
    series['LowQoS'] = qospref[0]
    series['HighQoS'] = qospref[1]
    return series

def load_planetlab():
    ldata = np.loadtxt('eua/PlanetLabData_1')[np.tril_indices(490)]
    ldata = ldata[ldata != 0]
    ldata = np.unique(ldata)
    latency_row = 150
    latency_col = (ldata.shape[0] // latency_row)
    ldata = np.resize(ldata, latency_col * latency_row)
    latency = ldata.reshape(latency_row, -1)
    return latency

def fetch_network_lat(distance, latency_data):
    rep_lat = np.random.choice(latency_data[distance], size=1, replace=True)
    return rep_lat / 1000

def load_users(num_of_users):
    user_raw = pd.read_csv("eua/users.csv").rename_axis("UID")
    df = user_raw.sample(num_of_users)
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude), crs='epsg:4326')
    user = gdf[['geometry']].to_crs(epsg=28355)
    return user

def load_servers(num_of_servers):
    server_raw = pd.read_csv("eua/servers.csv").rename_axis("SID")
    df = server_raw.sample(num_of_servers)
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.LONGITUDE, df.LATITUDE), crs='epsg:4326')
    server = gdf[['geometry']].to_crs(epsg=28355)
    def add_radius(series):
        radius = 150
        series.geometry = series.geometry.buffer(radius)
        series['radius'] = radius
        return series
    server = server.apply(add_radius, axis=1)
    return server

def ngb_matrix(U, N, S):
    user = load_users(U)
    server = load_servers(N)
    neighbourhood = np.zeros([U, N])
    network_latency = np.zeros(N)
    latency_data = load_planetlab()
    for u in range(U):
        for n in range(N):
            if server.iloc[n].geometry.contains(user.iloc[u].geometry):
                neighbourhood[u, n] = 1
                distance = server.iloc[n].geometry.centroid.distance(user.iloc[u].geometry)
                rep_lat = fetch_network_lat(int(distance), latency_data)
                if network_latency[n] < rep_lat:
                    network_latency[n] = rep_lat
            else:
                neighbourhood[u, n] = 0
    service = np.zeros(U)
    for u in range(U):
        service[u] = random.randrange(0, S, 1)
    server_service = np.zeros((N, S))
    for n in range(N):
        for u in range(U):
            if neighbourhood[u][n] == 1:
                server_service[n][int(service[u])] += 1
    return neighbourhood, user, server, service, server_service, network_latency

def generate_server_state(num_server, server_service, filename_base="dataset/dual_s_base.csv"):
    df = pd.read_csv(filename_base)
    df['workload_gpu'] = df['workload_gpu'].multiply(1/80).round(0).astype(int)
    ram = df.ram.unique()
    cores = df.cores.unique()
    workload_cpu = df.workload_cpu.unique()
    server_state = []
    gamma = []
    for s_id in range(num_server):
        gram = np.random.choice(ram, 1)[0]
        gcores = np.random.choice(cores, 1)[0]
        gwl_c = np.random.choice(workload_cpu, 1)[0]
        fetch_state = df.loc[(df['ram'] == gram) & (df['cores'] == gcores) & (df['workload_cpu'] == gwl_c)]
        gwl_g = fetch_state.sample().iloc[0]['workload_gpu']
        fetch_time = fetch_state.loc[(df['workload_gpu'] == gwl_g)]
        time_yolo = fetch_time['time_yolo'].mean()
        time_mnet = fetch_time['time_mnet'].mean()
        gs1 = server_service[s_id][0]
        gs2 = server_service[s_id][1]
        server_state.append([gram, gcores, gwl_c, gwl_g, gs1, gs2])
        gamma.append((time_yolo, time_mnet))
    return server_state, gamma

def ilp_algo(U, N, ngb, gamma, service, network_latency, latency_threshold):
    I = range(U)
    J = range(N)
    alloc = Model(sense=MAXIMIZE, name="alloc", solver_name=CBC)
    def coverage(user_ix, server_ix):
        return 1 if ngb[user_ix][server_ix] == 1 else 0
    x = [[alloc.add_var(f"x{i}{j}", var_type=BINARY) for j in J] for i in I]
    alloc.objective = xsum(x[i][j] for i in I for j in J)
    for i in I:
        for j in J:
            if not coverage(i, j):
                alloc += x[i][j] == 0
    for i in I:
        alloc += xsum(x[i][j] for j in J) <= 1
    for j in J:
        alloc += xsum(gamma[j][int(service[i])] * x[i][j] for i in I) <= latency_threshold - network_latency[j]
    alloc.optimize(max_seconds=25)
    return [(i, j) for i in I for j in J if x[i][j].x >= 0.99]

def rl_algo(U, N, S, ngb, service, server_state, model_rl):
    server_capacity = np.zeros((N, S))
    for server_id in range(N):
        state = server_state[server_id]
        action = model_rl.predict(np.array(state), deterministic=True)
        u1 = action[0] // 5 + 1
        u2 = (action[0] + 1) - (u1 - 1) * 5
        server_capacity[server_id][0] = u1 * 100
        server_capacity[server_id][1] = u2 * 100
    rl_allocation = []
    for i in range(U):
        server_list = np.where(ngb[i, :N] == 1)[0]
        if len(server_list) == 0:
            continue
        ser = int(service[i])
        choosen_server = server_list[np.argmax(server_capacity[server_list, ser])]
        if server_capacity[choosen_server][ser] > 0:
            server_capacity[choosen_server][ser] -= 1
            rl_allocation.append((i, choosen_server))
    return rl_allocation

def greedy_algo(U, N, ngb, user, server, service, gamma, latency_threshold, network_latency):
    server_capacity = np.zeros(N)
    allocations = []
    for user_id in range(U):
        server_ngb_list = np.where(ngb[user_id, :N] == 1)[0]
        if len(server_ngb_list) == 0:
            continue
        dist_list = np.array([
            server_ngb_list,
            [server.iloc[i]['geometry'].centroid.distance(user.iloc[user_id]['geometry']) for i in server_ngb_list]
        ])
        sorted_distance_list = dist_list[:, dist_list[1].argsort()]
        server_list = sorted_distance_list[0].astype(int)
        for server_id in server_list:
            lat = gamma[server_id][int(service[user_id])]
            if server_capacity[server_id] + lat <= latency_threshold - network_latency[server_id]:
                server_capacity[server_id] += lat
                allocations.append((user_id, server_id))
                break
    return allocations

def heft_algo(U, N, ngb, user, server, service, gamma, latency_threshold, network_latency):
    user_priorities = []
    for u in range(U):
        server_ngb_list = np.where(ngb[u, :N] == 1)[0]
        if len(server_ngb_list) == 0:
            continue
        avg_time = np.mean([gamma[sid][int(service[u])] + network_latency[sid] for sid in server_ngb_list])
        user_priorities.append((u, avg_time))
    user_priorities.sort(key=lambda x: x[1])

    server_capacity = np.zeros(N)
    allocations = []
    for (u, _) in user_priorities:
        server_ngb_list = np.where(ngb[u, :N] == 1)[0]
        best_server = None
        best_finish_time = float("inf")
        for sid in server_ngb_list:
            finish_time = server_capacity[sid] + gamma[sid][int(service[u])] + network_latency[sid]
            if finish_time <= latency_threshold and finish_time < best_finish_time:
                best_finish_time = finish_time
                best_server = sid
        if best_server is not None:
            server_capacity[best_server] += gamma[best_server][int(service[u])]
            allocations.append((u, best_server))
    return allocations

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model_rl = DQN.load(args.model_path)
    result_columns = ["user", "ilp_user", "greedy_user", "heft_user", "rl_user"]
    result_df = pd.DataFrame(columns=result_columns)

    for U in range(400, 800, 100):
        for epoch in range(args.episodes):
            ngb, user, server, service, server_service, network_latency = ngb_matrix(U, args.N, args.S)
            server_state, gamma = generate_server_state(args.N, server_service)

            ilp_alloc = ilp_algo(U, args.N, ngb, gamma, service, network_latency, args.latency_threshold)
            greedy_alloc = greedy_algo(U, args.N, ngb, user, server, service, gamma, args.latency_threshold, network_latency)
            heft_alloc = heft_algo(U, args.N, ngb, user, server, service, gamma, args.latency_threshold, network_latency)
            rl_alloc = rl_algo(U, args.N, args.S, ngb, service, server_state, model_rl)

            result_df.loc[len(result_df)] = [U, len(ilp_alloc), len(greedy_alloc), len(heft_alloc), len(rl_alloc)]
            print(f"[Users={U}] ILP={len(ilp_alloc)}, Greedy={len(greedy_alloc)}, HEFT={len(heft_alloc)}, RL={len(rl_alloc)}")

    result_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

    plt.figure(figsize=(10, 6))
    for algo in ["ilp_user", "greedy_user", "heft_user", "rl_user"]:
        plt.plot(result_df["user"], result_df[algo], marker='o', label=algo.split("_")[0].upper())
    plt.xlabel("Number of Users")
    plt.ylabel("Allocated Users")
    plt.title("Allocation Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency_threshold", type=float, default=50)
    parser.add_argument("--S", type=int, default=2)
    parser.add_argument("--N", type=int, default=70)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="allocation_results/AAnew_threats_und_results.csv")
    parser.add_argument("--model_path", type=str, default="saved_models_Round1/und_agent_candidate_1.zip")
    args = parser.parse_args()
    main(args)
