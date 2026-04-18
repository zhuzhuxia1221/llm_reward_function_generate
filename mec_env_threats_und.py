import gym
from gym import spaces
import numpy as np
import random
import pandas as pd
import geopandas
import os

class MECEnvThreatsUnd(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_users=100, num_servers=50, num_services=2, latency_threshold=50):
        super(MECEnvThreatsUnd, self).__init__()
        self.num_users = num_users
        self.num_servers = num_servers
        self.num_services = num_services
        self.latency_threshold = latency_threshold
        self.users_df = pd.read_csv("eua/users.csv")
        self.servers_df = pd.read_csv("eua/servers.csv")
        self.dual_s_base = pd.read_csv("dataset/dual_s_base.csv")
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        self.reset()
        self.df = self.dual_s_base

    def _load_planetlab(self):
        ldata = np.loadtxt('eua/PlanetLabData_1')[np.tril_indices(490)]
        ldata = ldata[ldata != 0]
        ldata = np.unique(ldata)
        latency_row = 150
        latency_col = (ldata.shape[0] // latency_row)
        ldata = np.resize(ldata, latency_col * latency_row)
        return ldata.reshape(latency_row, -1)

    def _fetch_network_lat(self, distance, latency_data):
        rep_lat = np.random.choice(latency_data[distance], size=1, replace=True)
        return rep_lat / 1000

    def _generate_network_state(self):
        user_sample = self.users_df.sample(self.num_users)
        server_sample = self.servers_df.sample(self.num_servers)

        user_gdf = geopandas.GeoDataFrame(
            user_sample,
            geometry=geopandas.points_from_xy(user_sample.Longitude, user_sample.Latitude),
            crs='epsg:4326'
        ).to_crs(epsg=28355)

        server_gdf = geopandas.GeoDataFrame(
            server_sample,
            geometry=geopandas.points_from_xy(server_sample.LONGITUDE, server_sample.LATITUDE),
            crs='epsg:4326'
        ).to_crs(epsg=28355)

        server_gdf['geometry'] = server_gdf['geometry'].buffer(150)

        neighbourhood = np.zeros((self.num_users, self.num_servers))
        network_latency = np.zeros(self.num_servers)
        latency_data = self._load_planetlab()

        for u in range(self.num_users):
            for n in range(self.num_servers):
                if server_gdf.iloc[n].geometry.contains(user_gdf.iloc[u].geometry):
                    neighbourhood[u, n] = 1
                    distance = server_gdf.iloc[n].geometry.centroid.distance(user_gdf.iloc[u].geometry)
                    rep_lat = self._fetch_network_lat(int(distance), latency_data)
                    if network_latency[n] < rep_lat:
                        network_latency[n] = rep_lat

        service_assignment = np.random.randint(0, self.num_services, size=self.num_users)

        server_service_count = np.zeros((self.num_servers, self.num_services))
        for n in range(self.num_servers):
            for u in range(self.num_users):
                if neighbourhood[u, n] == 1:
                    server_service_count[n][service_assignment[u]] += 1

        return neighbourhood, service_assignment, server_service_count, network_latency

    def _generate_server_state(self, server_id):
        df = self.dual_s_base.copy()
        df['workload_gpu'] = df['workload_gpu'].multiply(1/80).round(0).astype(int)

        ram_choice = np.random.choice(df.ram.unique(), 1)[0]
        core_choice = np.random.choice(df.cores.unique(), 1)[0]
        workload_cpu_choice = np.random.choice(df.workload_cpu.unique(), 1)[0]

        fetch_state = df.loc[
            (df['ram'] == ram_choice) &
            (df['cores'] == core_choice) &
            (df['workload_cpu'] == workload_cpu_choice)
        ]

        workload_gpu_choice = fetch_state.sample().iloc[0]['workload_gpu']

        service1_count = self.server_service_count[server_id][0]
        service2_count = self.server_service_count[server_id][1]

        return [ram_choice, core_choice, workload_cpu_choice,
                workload_gpu_choice, service1_count, service2_count]

    def reset(self):
        """Reset environment"""
        self.ngb_matrix, self.service_assignment, self.server_service_count, self.network_latency = self._generate_network_state()
        self.current_server = 0
        self.state = self._generate_server_state(self.current_server)
        return np.array(self.state, dtype=np.float32)

    def get_reward(self, state, action):
        u1 = (action // 5) * 4 + 1
        u2 = (action % 5) * 4 + 1

        gram, gcores, gwl_c, gwl_g, gs1_old, gs2_old = state

        df = self.dual_s_base
        fetch_state = df.loc[
            (df['ram'] == gram) &
            (df['cores'] == gcores) &
            (df['workload_cpu'] == gwl_c) &
            (df['workload_gpu'] == gwl_g) &
            (df['users_yolo'] == u1) &
            (df['users_mnet'] == u2)
            ]

        if fetch_state.empty:
            return -20

        time1 = fetch_state.sample().iloc[0]['time_yolo']
        time2 = fetch_state.sample().iloc[0]['time_mnet']
        tm = max(time1, time2)

        if tm <= self.latency_threshold:
            return (u1 - gs1_old) + (u2 - gs2_old) + u1 + u2
        else:
            return -u1 - u2

    def step(self, action):
        reward = self.get_reward(self.state, action)

        done = False
        self.current_server += 1
        if self.current_server >= self.num_servers:
            done = True
        else:
            self.state = self._generate_server_state(self.current_server)

        return np.array(self.state, dtype=np.float32), reward, done, {}


