import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax


class SACAgent:
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        '''        

        np.tile: Replicates prior distribution across batch dimension

        Creates [batch_size x n_skills] matrix for vectorized operations

        Example: If p_z = [0.2, 0.8] and batch_size=32 → 32x2 matrix

        '''
        #print("SACAgent initialized with p_z:", self.p_z)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],
                                            action_bounds=self.config["action_bounds"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        #print("Policy network initialized with device:", self.device)
        # Policy network is initialized with the device (CPU or GPU) based on availability
        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        #print("Q-value network 1 initialized with device:", self.device)
        # Q-value network 1 is initialized with the device (CPU or GPU) based on availability
        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        #print("Q-value network 2 initialized with device:", self.device)
        # Q-value network 2 is initialized with the device (CPU or GPU) based on availability
        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        #print("Value network initialized with device:", self.device)
        # Value network is initialized with the device (CPU or GPU) based on availability
        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        #print("Value target network initialized with device:", self.device)
        self.hard_update_target_network()
        #print("Hard update target network completed.")
        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        #print("Discriminator initialized with device:", self.device)
        # Discriminator is initialized with the device (CPU or GPU) based on availability
        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, states):
        #print("FUNCTION Choosing action for states:", states)
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)   #sample_or_likelihood: Returns action + log probability
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state): #Converts all components to CPU tensors
        #print("FUNCTION Storing transition with state:", state, "z:", z, "done:", done, "action:", action, "next_state:", next_state)
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)  #Stores in replay buffer

    def unpack(self, batch):   #Batch Unpacking
        #print("FUNCTION Unpacking batch:", batch)
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)  #Concatenates tensors along batch dimension, Reshapes tensors to expected network input shapes
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        print("FUNCTION Training agent with current memory size:", len(self.memory))
        if len(self.memory) < self.batch_size:  #Checks if enough samples in buffer
            return None
        else:
            #print("Training with batch size:", self.batch_size)
        # print("Sampling random batch from memory...")
            batch = self.memory.sample(self.batch_size)   #Samples random batch from replay memory
            states, zs, dones, actions, next_states = self.unpack(batch)  #Unpacks and formats tensors
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            #print("Calculating value target...")
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)   #Samples actions with noise 
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)   #Takes minimum Q-value for conservative estimate
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()
            #print("Calculating value loss...")
            #Calculating the value loss
            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)
            #The discriminator is trained to predict z from next_state
            #print("Calculating intrinsic rewards...")
            # Discriminator predicts skill from next state
            logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])  #torch.split: Separates state and skill components
            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            '''
            The intrinsic reward is:
                rintrinsic=log⁡qϕ(z∣s)−log⁡p(z)
                rintrinsic​=logqϕ​(z∣s)−logp(z)

                Since p(z)p(z) is uniform, this simplifies to maximizing log⁡qϕ(z∣s)logqϕ​(z∣s).

                SAC updates its policy to maximize this reward and entropy.
                So now the agent is being rewarded for:

                Creating states that help the discriminator identify z.

                Being uncertain (stochastic) enough to explore.

                It’s like saying: “Do something only Skill 2 would do, and do it creatively.”
            '''
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)   #Discriminator Reward --- INTRINSIC REWARDS
            #print("Rewards calculated:", rewards)
            # Calculating the Q-Value target
            with torch.no_grad():  #Disables gradient tracking
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states, actions) #Computes TD errors for both Q-networks
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)  #Computes TD errors for both Q-networks
            q2_loss = self.mse_loss(q2, target_q)
            #print("Q-value losses calculated:", q1_loss.item(), q2_loss.item())
            policy_loss = (self.config["alpha"] * log_probs - q).mean()    #Maximizes entropy (log_probs) while maximizing Q-values ---Policy Optimization
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0]) #Cross-entropy between predicted and actual skills  #Trains discriminator to recognize skill from states
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))
            #print("Policy loss:", policy_loss.item(), "Discriminator loss:", discriminator_loss.item())
            #print("Discriminator logits:", logits, "Discriminator zs:", zs.squeeze(-1))
            # Update the networks
            self.policy_opt.zero_grad() #Clears existing gradients
            policy_loss.backward()  # Computes gradients via autograd
            self.policy_opt.step() #Updates parameters using Adam

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()
            #print("Networks updated successfully.")
            # Update the target value network
            self.soft_update_target_network(self.value_network, self.value_target_network)
            #print("Target value network updated with soft update.")
            return -discriminator_loss.item()
 
    def soft_update_target_network(self, local_network, target_network):  #Polyak averaging: θ' ← τθ + (1-τ)θ'   Maintains stable target values during learning
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):  #Preserves random states for exact reproducibility
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)  #Handles both PyTorch and Python's random states

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
