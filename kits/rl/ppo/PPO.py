from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, MultivariateNormal

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dims: List[int], 
    ):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        
        self.n_actors = len(action_dims)
        
        actor_heads_dict = dict()
        for i, action_dim in enumerate(action_dims):
            actor_heads_dict[f'actor_{i}'] = nn.Sequential(    
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        self.actor_heads = nn.ModuleDict(actor_heads_dict)
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        intermediate_actor_features: Tensor = self.actor(state)
        
        actions: List[Tensor] = []
        action_logprobs: List[Tensor] = []
        
        for i in range(self.n_actors):
            actor_head = self.actor_heads[f'actor_{i}']
            action_probs: Tensor = actor_head(intermediate_actor_features)
            dist = Categorical(action_probs)
            action: Tensor = dist.sample()
            action_logprob: Tensor = dist.log_prob(action)
            
            actions.append(action.detach())
            action_logprobs.append(action_logprob.detach())
        
        state_val: Tensor = self.critic(state)

        return torch.Tensor(actions), torch.Tensor(action_logprobs), state_val.detach()
    
    def evaluate(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        intermediate_actor_features: Tensor = self.actor(state)
        
        action_logprobs: List[Tensor] = []
        dist_entropy_lst: List[Tensor] = []
        
        for i in range(self.n_actors):
            actor_head = self.actor_heads[f'actor_{i}']
            action_probs: Tensor = actor_head(intermediate_actor_features)
            dist = Categorical(action_probs)
            action_logprob: Tensor = dist.log_prob(action[:, i])
            dist_entropy: Tensor = dist.entropy()
            
            action_logprobs.append(action_logprob)
            dist_entropy_lst.append(dist_entropy)
        
        state_values = self.critic(state)
        
        return torch.stack(action_logprobs, axis=1), state_values, torch.stack(dist_entropy_lst, axis=1)


class PPO:
    def __init__(
        self, 
        state_dim: int,
        action_dims: List[int],
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dims).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dims).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        return action.numpy()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = torch.unsqueeze(rewards.detach() - old_state_values.detach(), axis=1)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss: Tensor = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path: str) -> None:
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path: str) -> None:
        self.policy_old.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        ))
        self.policy.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        ))
