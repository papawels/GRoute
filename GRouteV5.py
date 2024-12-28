#! /usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from collections import deque
import gymnasium as gym

import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch.nn import Linear, ReLU
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from scipy.signal import medfilt

#DEFINITIONS ==========================================================================================

#Topology of network
def create_2d_grid(m, n):
    G = nx.DiGraph()

    #Nodes use single digit system
    for i in range(m):
        for j in range(n):
            G.add_node(i*n+j)  
    #Edges between neighboring nodes (up, down, left, right)
    for i in range(m):
        for j in range(n):
            currNodeIdx = i*n + j
            if i < m - 1:  #Add edge to the node to the right then back
                G.add_edge(currNodeIdx, currNodeIdx + n, label= f"({currNodeIdx},{currNodeIdx+n})", color="black")
                G.add_edge(currNodeIdx + n, currNodeIdx, color="black")
            if j < n - 1:  #Add edge to the node below then back
                G.add_edge(currNodeIdx, currNodeIdx + 1, label= f"({currNodeIdx},{currNodeIdx+1})", color="black")
                G.add_edge(currNodeIdx+1, currNodeIdx, color="black")
    return G
#Updates state space for gData
def update_state_space(data,node,fIdx,value):
    data.node_features[node,fIdx] = value
    return data

#Updates edge feature for gData and graph object
def update_edge_feature(grid,data,tranz,receive,value):

    grid[tranz][receive]['weight'] = value
    for i in range(230):
        u = eIdx[0][i].item()
        v = eIdx[1][i].item()
        if u == tranz and v == receive:
            data.edge_attr[i] = value
            print((u,v))
            print(i)
    return grid, data

#Updates only nFtrs (input for gData)
def update_nFtrs(nData,nodeNum,fIdx,value):
    nData[nodeNum][fIdx] = value
    return nData

#Updates edge color to visualize route followed
def update_edge_color(grid,tranz,receive,color):
    grid[tranz][receive]["color"] = color
    return grid

#Utility function (virtual sattellite nodes) 
def utility(z, alpha):
    u = (z ** (1 - alpha)) / ( 1 - alpha )
    return u

#Total utilty function (agent)
def req_utility(u_bw,u_d,lmbda):
    um = u_bw - lmbda * u_d;
    return um

#Will be used to smooth out spikes in data 
#def moving_avg(data, window_size):
#    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
def remove_spikes(data, threshold):
    result = data.copy()
    for i in range(1, len(data) - 1):
        if abs(data[i] - data[i-1]) > threshold and abs(data[i] - data[i+1]) > threshold:
            result[i] = (data[i-1] + data[i+1]) / 2
    return result

#CLASSES ===================================================================================================

#Class for each MPNN Layer 
class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')  #Aggregation = add

        #Fully connected neural network for the message function
        self.message_nn = nn.Sequential(
            nn.Linear(2 * in_channels + 1, 128),  #Concatenate src, dest, edge features
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

        #Fully connected neural network for the update function
        self.update_nn = nn.Sequential(
            nn.Linear(out_channels + in_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        #x = Node features (form = [num_nodes, in_channels])
        #edge_index = Graph connectivity (form = [2, num_edges])
        #edge_attr = Edge features (form = [num_edges, edge_features])
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        #x_i =  Features of target nodes (form = [num_edges, in_channels])
        #x_j = Features of source nodes (form = [num_edges, in_channels])
        #edge_attr = Edge features (form = [num_edges, edge_features])

        #Concatenate source, target, and edge features
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_nn(message_input)

    def update(self, aggr_out, x):
        #aggr_out = Aggregated messages (form = [num_nodes, out_channels])
        #x = Original node features (form = [num_nodes, in_channels])
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_nn(update_input)

#3 layered fully conencted Message Passing Neural Network Object Class
class MPNN(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, num_classes):
        super(MPNN, self).__init__()
        self.mpnn1 = MPNNLayer(node_in_channels, hidden_channels) #layer for known links with hidden
        self.mpnn2 = MPNNLayer(hidden_channels, hidden_channels) #layer for hidden with hidden

        #Readout function: Aggregates node features to a graph-level representation
        self.readout_nn = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        #batch: Batch indices for each node (form = [num_nodes])

        #Message Passing
        x = self.mpnn1(x, edge_index, edge_attr)
        x = self.mpnn2(x, edge_index, edge_attr)

        #Graph-level pooling (sum pooling)
        graph_features = scatter(x, batch, dim=0, reduce='add')

        #Readout
        return self.readout_nn(graph_features)

#Routing Environment
class RoutingEnv:
    def __init__(self, graph, source, target):
        self.graph = graph
        self.source = source
        self.target = target
        self.reset()
    
    #Start from source again
    def reset(self):
        self.current_node = self.source
        self.visited = set([self.source])
        return self.current_node

    #action = next node
    def step(self, action):
        if action in self.visited:
            reward = -50  # Penalize revisiting nodes
            done = False
        elif action == self.target:
            numHops = len(self.visited) + 1 #Updates number of hops per action
            alphaDp = 0.9 #Delay variable for utility
            lmbda = 0.9 #Weight of delay utility penalty
            reward = -lmbda*utility(numHops,alphaDp)  #Penalty for how many hopes occured
            done = True
        else:
            cN = self.current_node
            reward = self.graph[cN][action]['weight']  # Utility per hop
            self.visited.add(action)
            done = False

        self.current_node = action

        betweenTmp = nFtrs[action][2]
        newNFtrs = update_nFtrs(nFtrs,action,2,betweenTmp+1) #Betweeness increase by one very time node is used every episodes
        newNFtrs = update_nFtrs(newNFtrs,action,3,1) #Action vector updated to 1 (used) from 0 (not used) after every episode

        return self.current_node, reward, done, newNFtrs

    #Chooses best neightbor to hop to
    def best_action(self):
        src = self.current_node
        actions = []
        eW = []
        #Look up table for possible neighbors
        for i in range(230):
            u = eIdx[0][i].item()
            v = eIdx[1][i].item()
            if u == src:
                #Action is only valid if bandwidth requirment is met
                if newNFtrs[v][0] >= BWreq:
                    actions.append(v)
                    eW.append(self.graph[u][v]['weight'])
        #Action with largest edge feature (weight) is chosen
        if len(eW) > 0:
            mW = max(eW)
            mWIdx = eW.index(mW)
            action = actions[mWIdx]
        #Just in case no actions work, use random choice
        else:
            action = random_choice()
        return action
    
    #Random neighbor chosen to hop to
    def random_choice(self):
        src = self.current_node
        cActs = []
        eW = []
        #Look up table for possible neighbors
        for i in range(230):
            u = eIdx[0][i].item()
            v = eIdx[1][i].item()
            if u == src:
                cActs.append(v)
                eW.append(self.graph[u][v]['weight'])
        #Select what is left 
        if (len(cActs)-1) == 0:
            aIdx = 0
        #Select randomly one possible neighbor
        else:
            aIdx = random.randint(0,len(cActs)-1)
        act = cActs[aIdx]
        return act

#VARIABLES & INITIALIZATION ============================================================================

m, n = 11, 6 #6 orbits with 11 satellites each
xs, ys = 2782700, 4051000 #Real Rounded X and Y spacing of Iridium constellation satellites
numNodes = m*n #Number of nodes

G = create_2d_grid(m, n) #2D grid (2D tensor)
G2 = nx.grid_2d_graph(m, n) #2D grid using 3D tensor for postion vector

numFtrs = 4 #Number of features
eIdx = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous() #Convert to PyG format
nFtrs = torch.zeros((numNodes, numFtrs))  #66 nodes, 4 features(f) each, initialized as zero
#f1 = Available capacity 
#f2 = Occupied bandwidth 
#f3 = Link Betweeness
#f4 = Action vector (use or no use)

BWs = [16,32,64] #Several packages of BW 
BWreqIdx = random.randint(0,2) #Randomly selected 
BWreq = BWs[BWreqIdx] #Selected BW of package for request
print("The BW requirment for this request is", BWreq,"Mbps")

#Parameters for rewards/utility
alphaBW = 0.9 
alphaDp = 0.9
lmbda = 1
uBWm = utility(BWreq,alphaBW) #utility bandwidth

#Assign random weights to edges iitialized to uBWm then added randomness for simulated traffic 
#and envirnomental conditions
for u, v in G.edges():
    G[u][v]['weight'] = uBWm*random.uniform(0.5,1)  

numLinks = len(G.edges) #Number of links
eFtrs = torch.zeros(numLinks,1) #230 edges, 1 feature each, initialized as zero
#Syncronizes eFtrs and graph weights to be consistent
for i in range(numLinks):
    u = eIdx[0][i].item()
    v = eIdx[1][i].item()
    w = G[u][v]['weight']
    eFtrs[i] = w

#eFtrs = torch.zeros(115*2,1) #230 edges, 1 feature each, initialized as zero 
batch = torch.zeros(numNodes, dtype=torch.long) #All nodes belong to a single graph
gData = Data(node_features=nFtrs, edge_index=eIdx, edge_attr=eFtrs) #Tensor dataset

snrLow = 5 #Simulated worse SNR (realisitc)
snrHigh = 20 #Simulated best SNR (realistic)
initBW = 200 #Initilaized BW available 
trafLow = 100 #Low traffic 
trafHigh = 125 #High traffic

#Adding traffic conditions to the nodes
for i in range(numNodes):
    SNR = random.randint(snrLow,snrHigh)
    occBW = random.randint(trafLow,trafHigh) #Simulating random traffic
    cAvail = (initBW-occBW) * math.log(1+SNR,2) #Available capcity
    update_state_space(gData,i,1,occBW) #Updating f2
    update_state_space(gData,i,0,cAvail) #Updating f1

#Selecting the source and destination nodes
src = random.randint(0,5) #Random Source (nodes 0 to 5)
dest = random.randint(60,65) #Random Destination (nodes 60 to 65)
print("The Source Node will be node", src)
print("The Destination Node will be node", dest)

#4 node features, 1 edge feature, 20 output feature size, binary classification
model = MPNN(node_in_channels=numFtrs, edge_in_channels=1, hidden_channels=60, num_classes=2) 
output = model(nFtrs, eIdx, eFtrs, batch)

#MACHINE-LEARNING GROUTUNG =================================================================================

#Hyper parameters 
gamma = 0.9 #Discount factor
eps = 1.0 #Starting exploration rate
eMin = 0.001 #Min exploration rate
eD = 0.95 #Exploration rate decay
lR = 0.001 #Learning Rate
rPS = 300 #Replay Pool Size
numEpisodes = 1000
batch_size = 100

#Intialize to base model
newNFtrs = nFtrs

#Initialized containers
rewards = []
hops = []
tput = []
avgTput = []

#DQN Based ML model
env = RoutingEnv(G,src,dest)
optimizer = Adam(model.parameters(), lr=lR)
rBuff = deque(maxlen=rPS)
#Per episode is a route
for episode in range(numEpisodes):
    state = env.reset() #Environment is reset every episode
    total_reward = 0 #Reward is reset every episode
    action_list = []
    action_list.append(src) #Counting source as part of action list
    
    #Initialize Action vector as zero since its episode dependent
    for i in range(numNodes):
        newNFtrs = update_nFtrs(newNFtrs,i,3,0)

    while True:
        #Compute Q-values
        with torch.no_grad():
            output = model(newNFtrs, eIdx, eFtrs, batch)
            q_values = output
        #Epsilon-greedy action selection
        #Allowing some exploratory actions
        if random.random() < eps:
            action = env.random_choice()
        else:
            action = env.best_action()
        
        #Try to promote moving forward if agent keeps looping same nodes
        if action in action_list:
            dEnd = (numNodes - 1 - env.current_node)
            dDest = (dest - env.current_node)
            if dEnd > 5:
                action = env.current_node + 6
            else:
                if dDest < 0:
                    action = env.current_node - 1
                else:
                    action = env.current_node + 1
        
        tput.append(nFtrs[action][0]) #Adding Avail capacity from sucessful action
        action_list.append(action)
        #print(action_list) #Prining Node Route List (Src to Dest Path) - Debugging
        #Step in environment
        next_state, reward, done, newNFtrs = env.step(action)
        rBuff.append((state, action, reward, next_state, done))

        #Train the model, when the buffer fill, it will sample older data and slowly overwrite
        if len(rBuff) > rPS:
            batch = random.sample(rPS, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.long)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.long)
            dones = torch.tensor(dones, dtype=torch.float32)

            current_q_values = model(newNFtrs, gData.edge_index, states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = model(newNFtrs, gData.edge_index, next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = (current_q_values - target_q_values.detach()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward += reward
        state = next_state

        if done:
            break
    #Exploratory rate decreases per episodes to help converge on an answer
    #Decreases every third episode
    if episode % 3 == 0:
        eps = max(eMin, eps*eD)
    #print(action_list)
    tmpTput = np.mean(tput) #finding average throughput of route this episode
    avgTput.append(tmpTput) #minimum avail capcity per episode
    hops.append(len(action_list)) #number of hops per episode
    rewards.append(total_reward)
    #print(f"Episode {episode}, Total Reward: {total_reward}")
    
gRouteActions = action_list

smoothReward = medfilt(rewards, kernel_size=5)
ex = range(numEpisodes)
plt.plot(ex,smoothReward)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total Reward per Episode for GRouting Method")
plt.show()

smoothDelay = medfilt(hops, kernel_size=5)
plt.plot(ex,hops)
plt.xlabel("Episode")
plt.ylabel("Delay(Hops)")
plt.title("Number of Hops per Episode for GRouting Method")
plt.show()

plt.plot(ex,avgTput)
plt.xlabel("Episode")
plt.ylabel("Throughput (BW Units)")
plt.title("Throughput per Episode for GRouting Method")
plt.show()

#PLOTTING TOPOLOGY GRoute =================================================================================

#Colors for all nodes
nColors = []
for node in G.nodes():
    if node in gRouteActions:
        if node == src or node == dest:
            nColors.append("dodgerblue")
        else:
            nColors.append("lawngreen")
    else:
        nColors.append("lightblue")


edge_colors = [G[u][v]["color"] for u, v in G.edges()] #Colors for all nodes

plt.figure(figsize=(8, 6),num='Iridium Constellation Optimal Route') #Figure itself
pos = { (i*n+j):(i*xs, j*ys) for i, j in G2.nodes()} #Position of nodes on graph
#Drawing the graph on the figure
nx.draw(G, pos, with_labels=True, node_size=600, node_color=nColors, font_weight='bold', font_size=14, edge_color=edge_colors)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()

#MACHINE-LEARNING SHORTEST PATH ==========================================================================
#Same model as above but focusing on using shortest path logic rather than best_action

#Initiallized to start the same way as previous model
newNFtrs = nFtrs
rewards = []
hops = []
tput = []
avgTput = []

env = RoutingEnv(G,src,dest)
optimizer = Adam(model.parameters(), lr=lR)
rBuff = deque(maxlen=rPS)
for episode in range(numEpisodes):
    state = env.reset()
    total_reward = 0
    action_list = []
    action_list.append(src)

    for i in range(numNodes):
        newNFtrs = update_nFtrs(newNFtrs,i,3,0)

    while True:
        # Compute Q-values
        with torch.no_grad():
            q_values = output
        # Epsilon-greedy action selection
        if random.random() < eps:
            action = env.random_choice()
        else:
            #Shortest path logic (not best_action)
            dEnd = (numNodes - 1 - env.current_node)
            dDest = (dest - env.current_node)
            if dEnd > 5:
                action = env.current_node + 6
            else:
                if dDest < 0:
                    action = env.current_node - 1
                else:
                    action = env.current_node + 1
        
        tput.append(nFtrs[action][0])
        action_list.append(action)
        # Step in environment
        next_state, reward, done, newNFtrs = env.step(action)
        rBuff.append((state, action, reward, next_state, done))

        # Train the model
        if len(rBuff) > rPS:
            batch = random.sample(rPS, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.long)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.long)
            dones = torch.tensor(dones, dtype=torch.float32)

            current_q_values = model(newNFtrs, gData.edge_index, states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = model(newNFtrs, gData.edge_index, next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = (current_q_values - target_q_values.detach()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward += reward
        state = next_state

        if done:
            break

    if episode % 3 == 0:
        eps = max(eMin, eps*eD)
    #print(action_list)
    tmpTput = np.mean(tput) 
    avgTput.append(tmpTput)
    hops.append(len(action_list))
    rewards.append(total_reward)
    #print(f"Episode {episode}, Total Reward: {total_reward}")

sDistActions = action_list

plt.plot(ex,rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total Reward per Episode for Shortest Path")
plt.show()

plt.plot(ex,hops)
plt.xlabel("Episode")
plt.ylabel("Delay(Hops)")
plt.title("Number of Hops per Episode for Shortest Path")
plt.show()

plt.plot(ex,avgTput)
plt.xlabel("Episode")
plt.ylabel("Throughput (BW Units)")
plt.title("Throughput per Episode for Shortest Path")
plt.show()

#PLOTTING TOPOLOGY SHORTEST DISTANCE ======================================================================

#Colors for all nodes
nColors = []
for node in G.nodes():
    if node in sDistActions:
        if node == src or node == dest:
            nColors.append("dodgerblue")
        else:
            nColors.append("lawngreen")
    else:
        nColors.append("lightblue")


edge_colors = [G[u][v]["color"] for u, v in G.edges()] #Colors for all nodes

plt.figure(figsize=(8, 6),num='Iridium Constellation Optimal Route') #Figure itself
pos = { (i*n+j):(i*xs, j*ys) for i, j in G2.nodes()} #Position of nodes on graph
#Drawing the graph on the figure
nx.draw(G, pos, with_labels=True, node_size=600, node_color=nColors, font_weight='bold', font_size=14, edge_color=edge_colors)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()

#MACHINE-LEARNING RANDOM PATH ============================================================================
#Same model as above 2 but focusing on using only random choice 

newNFtrs = nFtrs
rewards = []
hops = []
tput = []
avgTput = []

env = RoutingEnv(G,src,dest)
optimizer = Adam(model.parameters(), lr=lR)
rBuff = deque(maxlen=rPS)
for episode in range(numEpisodes):
    state = env.reset()
    total_reward = 0
    action_list = []
    action_list.append(src)
    
    for i in range(numNodes):
        newNFtrs = update_nFtrs(newNFtrs,i,3,0)

    while True:
        # Compute Q-values
        with torch.no_grad():
            q_values = output
        #Only random choice (fully exploratory)
        action = env.random_choice()
        tput.append(nFtrs[action][0])
        action_list.append(action)
        #print(action_list)
        # Step in environment
        next_state, reward, done, newNFtrs = env.step(action)
        rBuff.append((state, action, reward, next_state, done))

        # Train the model
        if len(rBuff) > rPS:
            batch = random.sample(rPS, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.long)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.long)
            dones = torch.tensor(dones, dtype=torch.float32)

            current_q_values = model(newNFtrs, gData.edge_index, states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = model(newNFtrs, gData.edge_index, next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = (current_q_values - target_q_values.detach()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward += reward
        state = next_state

        if done:
            break

    if episode % 3 == 0:
        eps = max(eMin, eps*eD)
    #print(action_list)
    tmpTput = np.mean(tput) 
    avgTput.append(tmpTput)
    rewards.append(total_reward)
    hops.append(len(action_list))
    #print(f"Episode {episode}, Total Reward: {total_reward}")

ranActions = action_list

plt.plot(ex,rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total Reward per Episode using Random Path")
plt.show()

plt.plot(ex,hops)
plt.xlabel("Episode")
plt.ylabel("Delay(Hops)")
plt.title("Number of Hops per Episode using Random Path")
plt.show()

plt.plot(ex,avgTput)
plt.xlabel("Episode")
plt.ylabel("Throughput (BW Units)")
plt.title("Throughput per Episode for Random")
plt.show()

#PLOTTING TOPOLOGY RANDOM ==================================================================================

#Colors for all nodes
nColors = []
for node in G.nodes():
    if node in ranActions:
        if node == src or node == dest:
            nColors.append("dodgerblue")
        else:
            nColors.append("lawngreen")
    else:
        nColors.append("lightblue")


edge_colors = [G[u][v]["color"] for u, v in G.edges()] #Colors for all nodes

plt.figure(figsize=(8, 6),num='Iridium Constellation Optimal Route') #Figure itself
pos = { (i*n+j):(i*xs, j*ys) for i, j in G2.nodes()} #Position of nodes on graph
#Drawing the graph on the figure 
nx.draw(G, pos, with_labels=True, node_size=600, node_color=nColors, font_weight='bold', font_size=14, edge_color=edge_colors)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()
