import copy, time
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from marl.nets import ActorCriticPPO


# ============== Utility dataclasses ==============
@dataclass
class TrainingArgs:
    total_timesteps: int = int(2e5)
    max_steps: int = 50
    num_steps: int = 50
    num_epochs: int = 5
    minibatch_size: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_GAE: float = 0.95
    epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    seed: int = 42
    is_GAE: bool = True
    log_dir: str = "runs/ppo_nash_pz"
    save_path: str = "out/ppo_nash_history"


# =================================================
# DictRolloutBuffer for PettingZoo Parallel API
# =================================================
class DictRolloutBuffer:
    def __init__(self, agent_ids):
        self.agent_ids = list(agent_ids)
        self.states = {a: [] for a in self.agent_ids}
        self.actions = {a: [] for a in self.agent_ids}
        self.rewards = {a: [] for a in self.agent_ids}
        self.next_states = {a: [] for a in self.agent_ids}
        self.logprobs = {a: [] for a in self.agent_ids}
        self.values = {a: [] for a in self.agent_ids}
        self.dones = []

    def clear(self):
        self.__init__(self.agent_ids)


# =================================================
# Main PPO Nash Agent
# =================================================
class PPONashAgent:
    def __init__(self, env, args: TrainingArgs):
        self.args = args
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert hasattr(env, "possible_agents"), "Expect PettingZoo parallel_env."

        obs_dict, _ = env.reset(seed=args.seed)
        self.agent_order = list(env.possible_agents)
        self.num_agents = len(self.agent_order)

        # policy networks
        self.policy_nets = []
        for agent in self.agent_order:
            n_actions = env.action_space(agent).n
            obs_dim = int(np.prod(env.observation_space(agent).shape))
            self.policy_nets.append(ActorCriticPPO(obs_dim, n_actions).to(self.device))

        self.buffer = DictRolloutBuffer(self.agent_order)
        self.old_policy_nets = [copy.deepcopy(net) for net in self.policy_nets]
        self.optimizers = [optim.Adam(net.parameters(), lr=args.lr) for net in self.policy_nets]
        self.critic_loss = nn.MSELoss()
        self.writer = SummaryWriter(log_dir=args.log_dir)

        self.history = {"episode_returns": [], "loss_history": [], "timing": {}}
        self._last_obs = obs_dict
        self.timing = dict(env_loop_time=0, action_time=0,
                           ppo_update_time=0, ppo_forward_time=0,
                           ppo_backward_time=0, total_train_time=0)

    # ------------------------------------------------
    def _obs_to_tensor(self, obs, agent):
        return torch.tensor(np.asarray(obs, np.float32).flatten(), device=self.device)

    def _choose_actions(self, obs_dict):
        actions, logps, values = {}, {}, {}
        t0 = time.perf_counter()
        for i, agent in enumerate(self.agent_order):
            if agent not in self.env.agents:
                continue
            with torch.no_grad():
                x = self._obs_to_tensor(obs_dict[agent], agent)
                probs, value = self.policy_nets[i](x)
                dist = torch.distributions.Categorical(probs)
                a = dist.sample()
                actions[agent] = int(a.item())
                logps[agent] = dist.log_prob(a)
                values[agent] = value.squeeze(-1)
        self.timing["action_time"] += time.perf_counter() - t0
        return actions, logps, values

    # ------------------------------------------------
    def _compute_advantages(self, rewards, values, next_value, dones_mask):
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        R = torch.zeros(1, device=rewards.device)
        for t in reversed(range(T)):
            R = rewards[t] + self.args.gamma * R * (1 - dones_mask[t])
            returns[t] = R
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - values.squeeze(-1)
        return returns, advantages

    def _compute_GAE(self, rewards, values, next_value, dones_mask):
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=rewards.device)
        next_value = next_value.view(1, 1)
        vals = torch.cat([values, next_value], dim=0)
        for t in reversed(range(T)):
            delta = rewards[t] + self.args.gamma * vals[t+1]*(1-dones_mask[t]) - vals[t]
            gae = delta + self.args.gamma*self.args.lambda_GAE*(1-dones_mask[t])*gae
            advantages[t] = gae
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)
        returns = advantages + vals[:-1].squeeze(-1)
        return returns, advantages

    # ------------------------------------------------
    def update(self, global_step):
        import time
        t_up0 = time.perf_counter()

        dones_mask = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        actor_loss_list, critic_loss_list, entropy_list = [], [], []
        returns_per_agent, adv_per_agent = {}, {}

        # ----- precompute returns & advantages -----
        t_fwd0 = time.perf_counter()
        for i, agent in enumerate(self.agent_order):
            old_states = torch.stack(self.buffer.states[agent]).to(self.device)
            old_actions = torch.stack(self.buffer.actions[agent]).to(self.device)
            old_rewards = torch.stack(self.buffer.rewards[agent]).to(self.device)
            old_next_states = torch.stack(self.buffer.next_states[agent]).to(self.device)
            old_log_probs = torch.stack(self.buffer.logprobs[agent]).to(self.device)
            old_values = torch.stack(self.buffer.values[agent]).to(self.device)
            with torch.no_grad():
                nv = self.old_policy_nets[i](old_next_states[-1])[1].squeeze(-1)
            if self.args.is_GAE:
                rets, advs = self._compute_GAE(old_rewards, old_values.view(-1,1), nv, dones_mask)
            else:
                rets, advs = self._compute_advantages(old_rewards, old_values, nv, dones_mask)
            returns_per_agent[agent] = rets
            adv_per_agent[agent] = advs
        self.timing["ppo_forward_time"] += time.perf_counter()-t_fwd0

        # ----- PPO epochs -----
        for _ in range(self.args.num_epochs):
            t_fwd = time.perf_counter()
            batches = []
            for i, agent in enumerate(self.agent_order):
                s = torch.stack(self.buffer.states[agent]).to(self.device)
                a = torch.stack(self.buffer.actions[agent]).to(self.device)
                lp_old = torch.stack(self.buffer.logprobs[agent]).to(self.device)
                ret = returns_per_agent[agent].to(self.device)
                adv = adv_per_agent[agent].to(self.device)

                probs, values = self.policy_nets[i](s)
                dist = torch.distributions.Categorical(probs)
                lp_new = dist.log_prob(a)
                ratios = torch.exp(lp_new - lp_old.detach())
                s1 = ratios * adv
                s2 = torch.clamp(ratios, 1-self.args.epsilon, 1+self.args.epsilon)*adv
                actor_loss = -torch.min(s1, s2).mean()
                critic_loss = self.critic_loss(ret, values.squeeze(-1))
                entropy = dist.entropy().mean()
                loss = actor_loss + self.args.value_coef*critic_loss - self.args.entropy_coef*entropy
                batches.append((i, loss, actor_loss, critic_loss, entropy))
            self.timing["ppo_forward_time"] += time.perf_counter()-t_fwd

            t_bwd = time.perf_counter()
            for i, loss, al, cl, ent in batches:
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()
                actor_loss_list.append(al.item())
                critic_loss_list.append(cl.item())
                entropy_list.append(ent.item())
            self.timing["ppo_backward_time"] += time.perf_counter()-t_bwd

        self.timing["ppo_update_time"] += time.perf_counter()-t_up0
        return np.mean(actor_loss_list), np.mean(critic_loss_list), np.mean(entropy_list)

    # ------------------------------------------------
    def train(self):
        A = self.args
        episode_rewards, ep_ret = [], {a:0.0 for a in self.agent_order}
        obs = self._last_obs
        steps = 0
        start = datetime.now().replace(microsecond=0)
        train_t0 = time.perf_counter()

        while steps < A.total_timesteps:
            steps += 1
            acts, lps, vals = self._choose_actions(obs)
            t_env = time.perf_counter()
            nxt, r, term, trunc, _ = self.env.step(acts)
            done = all((term.get(a,False) or trunc.get(a,False)) for a in self.agent_order)
            for i,a in enumerate(self.agent_order):
                s  = obs.get(a, np.zeros(self.env.observation_space(a).shape, np.float32))
                ns = nxt.get(a, np.zeros_like(s))
                s_t, ns_t = self._obs_to_tensor(s,a), self._obs_to_tensor(ns,a)
                self.buffer.states[a].append(s_t)
                self.buffer.next_states[a].append(ns_t)
                if a in acts:
                    aa = torch.tensor(acts[a], dtype=torch.long, device=self.device)
                    lp, v = lps[a].to(self.device), vals[a].to(self.device)
                else:
                    aa = torch.tensor(0, dtype=torch.long, device=self.device)
                    lp = torch.tensor(0.0, device=self.device)
                    v  = torch.tensor(0.0, device=self.device)
                self.buffer.actions[a].append(aa)
                self.buffer.logprobs[a].append(lp)
                self.buffer.values[a].append(v)
                rew = float(r.get(a,0.0))
                self.buffer.rewards[a].append(torch.tensor(rew,device=self.device))
                ep_ret[a]+=rew
            self.buffer.dones.append(torch.tensor(float(done),device=self.device))
            self.timing["env_loop_time"] += time.perf_counter()-t_env
            obs = nxt
            if done:
                episode_rewards.append(ep_ret.copy())
                self.history["episode_returns"].append(ep_ret.copy())
                for ag in self.agent_order:
                    self.writer.add_scalar(f"{ag}/episode_return", ep_ret[ag], steps)
                ep_ret = {a:0.0 for a in self.agent_order}
                obs,_=self.env.reset(seed=A.seed)

            if steps%A.num_steps==0:
                aL,cL,eL=self.update(steps)
                self.buffer.clear()
                self.old_policy_nets=[copy.deepcopy(n) for n in self.policy_nets]
                self.history["loss_history"].append(
                    {"step":int(steps),"actor":float(aL),"critic":float(cL),"entropy":float(eL)}
                )
            if steps%500==0 and episode_rewards:
                print(f"[{steps}] last returns {episode_rewards[-1]} | elapsed {datetime.now().replace(microsecond=0)-start}")

        self.timing["total_train_time"]=time.perf_counter()-train_t0
        self.env.close(); self.writer.close()
        print(f"Training complete in {self.timing['total_train_time']:.1f}s")
        return episode_rewards
