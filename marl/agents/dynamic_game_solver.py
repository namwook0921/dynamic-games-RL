from dataclasses import dataclass
from torch.optim.lr_scheduler import StepLR
import numpy as np, json, copy
from datetime import datetime
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from gymnasium.spaces import Discrete
from marl.nets import ActorCriticPPO

# ----------------- Compact training args -----------------
@dataclass
class TrainingArgs:
    total_timesteps: int = int(1e5)
    max_steps: int = 500
    epsilon: float = 0.2
    gamma: float = 0.99
    lambda_GAE: float = 0.95
    lr: float = 3e-4
    grad_clip: float = 0.5
    num_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 64
    seed: int = 123
    is_GAE: bool = True
    action_sequence: list = None          # list[list[str]]; None => Nash (all in one tier)
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    log_steps: int = 5000
    log_dir: str = "runs/ppo_nash_pettingzoo"
    save_path: str = "training_history"   # saved as {save_path}.npz and {save_path}.json

# -------- dict-style rollout buffer keyed by agent id --------
class DictRolloutBuffer:
    def __init__(self, agent_ids):
        self.agent_ids = list(agent_ids)
        self.states      = {a: [] for a in self.agent_ids}
        self.actions     = {a: [] for a in self.agent_ids}
        self.rewards     = {a: [] for a in self.agent_ids}
        self.next_states = {a: [] for a in self.agent_ids}
        self.logprobs    = {a: [] for a in self.agent_ids}
        self.values      = {a: [] for a in self.agent_ids}
        self.dones = []  # per-step "episode done" (all agents terminated||truncated)
    def clear(self): self.__init__(self.agent_ids)

class DynamicGameSolver:
    def __init__(self, env, args: TrainingArgs | None = None, **legacy_kwargs):
        """
        Prefer passing a TrainingArgs; legacy kwargs still work for backward-compat.
        Uses args.action_sequence (list[list[str]]) to define decision tiers.
        Default (None) => Nash: all agents act simultaneously in one tier.
        """
        self.args = args or TrainingArgs(**legacy_kwargs)
        A = self.args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env

        assert hasattr(env, "possible_agents") and hasattr(env, "agents"), \
            "Expected PettingZoo env with .possible_agents and .agents"

        reset_out = env.reset(seed=A.seed)
        obs0, _ = (reset_out if isinstance(reset_out, tuple) else (reset_out, {}))
        if not isinstance(obs0, dict):
            raise ValueError("Use a Parallel API env (reset() -> (obs_dict, info)).")

        self.agent_order = list(env.possible_agents)
        self.num_agents = len(self.agent_order)
        if self.num_agents == 0:
            raise ValueError("Env has no possible_agents.")

        # ----- decision hierarchy (tiers) -----
        if A.action_sequence is None:
            self.turn_groups = [self.agent_order[:] ]
        else:
            self.turn_groups = A.action_sequence
            flat = [a for grp in self.turn_groups for a in grp]
            if set(flat) != set(self.agent_order) or len(flat) != len(self.agent_order):
                raise ValueError("args.action_sequence must be a partition of env.possible_agents with no duplicates.")

        self.flat_order = [a for grp in self.turn_groups for a in grp]
        self.agent_order = self.flat_order[:]
        self._index_of_agent = {ag: i for i, ag in enumerate(self.flat_order)}

        # earlier-by-agent: only agents in earlier tiers (not same tier)
        self._earlier_by_agent = {}
        seen = []
        for grp in self.turn_groups:
            for ag in grp:
                self._earlier_by_agent[ag] = list(seen)
            seen.extend(grp)

        # ----- build policy nets with hierarchy-aware input dims -----
        self.policy_nets = []

        for ag in self.agent_order:
            obs_shape = self.env.observation_space(ag).shape
            obs_dim = int(np.prod(obs_shape))

            extra = 0
            for ej in self._earlier_by_agent[ag]:
                sp = self.env.action_space(ej)
                if not isinstance(sp, Discrete):
                    raise TypeError(f"Only Discrete actions supported for augmentation; got {type(sp)} for {ej}")
                extra += sp.n

            in_dim = obs_dim + extra
            n_actions = self.env.action_space(ag).n
            self.policy_nets.append(ActorCriticPPO(in_dim, n_actions).to(self.device))
            print(f"[build] {ag}: obs_dim={obs_dim}, extra={extra}, in_dim={obs_dim+extra}, nA={n_actions}")

        self.trunks = []
        self.actor_heads = []
        self.critic_heads = []
        self.optimizers = []

        for net in self.policy_nets:
            # grab module parts
            trunk = getattr(net, "net") 
            actor_head = getattr(net, "actor_layer")
            critic_head = getattr(net, "critic_layer")

            self.trunks.append(trunk)
            self.actor_heads.append(actor_head)
            self.critic_heads.append(critic_head)

            # single optimizer, single lr = self.args.lr
            params_all = list(trunk.parameters()) \
                      + list(actor_head.parameters()) \
                      + list(critic_head.parameters())

            opt = torch.optim.Adam(params_all, lr=self.args.lr)
            self.optimizers.append(opt)

        self.old_policy_nets = [copy.deepcopy(net) for net in self.policy_nets]
        self.critic_loss = nn.MSELoss()

        self.buffer = DictRolloutBuffer(self.agent_order)
        self.writer = SummaryWriter(log_dir=A.log_dir)

        self._last_obs = obs0

        self.history = {
            "episode_returns": [],
            "loss_history": [],
        }

    # -------- utilities --------
    def _obs_to_tensor(self, obs, agent):
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _onehot(self, agent_id: str, actions_tensor: torch.Tensor) -> torch.Tensor:
        """One-hot encode actions for agent_id (scalar or [T])."""
        n = self.env.action_space(agent_id).n
        return torch.nn.functional.one_hot(actions_tensor.long(), num_classes=n).float().to(self.device)

    # ---------- single-step augmentation (used in action selection and bootstrap) ----------
    def _stackelberg_input_from_obs_and_prev_actions(self, obs_vec: torch.Tensor, prev_actions: dict, earlier_list: list[str]) -> torch.Tensor:
        """
        Concatenate one-hots for strictly earlier-tier agents present in prev_actions.
        earlier_list must be a list of agent IDs from self._earlier_by_agent[agent].
        """
        x = obs_vec if obs_vec.dim() == 1 else obs_vec.view(-1)
        for aj in earlier_list:
            n = self.env.action_space(aj).n
            if aj in prev_actions:
                oh = torch.nn.functional.one_hot(
                    torch.tensor(prev_actions[aj], device=self.device), num_classes=n
                ).float()
            else:
                oh = torch.zeros(n, device=self.device)
            x = torch.cat([x, oh], dim=-1)
        return x

    # ---------- sequence augmentation for PPO updates ----------
    def _augment_states(self, agent: str, states: torch.Tensor) -> torch.Tensor:
        """
        states: [T, obs_dim]; returns [T, obs_dim + sum(n_ej)] where ej are earlier-tier agents.
        Uses self.buffer.actions[ej] for each timestep's one-hot.
        """
        T = states.shape[0]
        extras = []
        for ej in self._earlier_by_agent[agent]:
            n = self.env.action_space(ej).n
            a_ej = torch.stack(self.buffer.actions[ej]).to(self.device).long()  # [T]
            oh = torch.nn.functional.one_hot(a_ej, num_classes=n).float()       # [T, n]
            extras.append(oh)
        if extras:
            extra_cat = torch.cat(extras, dim=-1)  # [T, sum n_ej]
            return torch.cat([states.to(self.device), extra_cat], dim=-1)
        return states.to(self.device)

    # ---------- action selection (leader -> follower within the same step) ----------
    def _choose_actions(self, obs_dict):
        actions, logps, values = {}, {}, {}
        live = set(self.env.agents)
        chosen_this_step = {}  # agent_id -> int action, in Stackelberg order

        for i, agent in enumerate(self.agent_order):
            if agent not in live:
                continue
            with torch.no_grad():
                base = self._obs_to_tensor(obs_dict[agent], agent)
                x = self._stackelberg_input_from_obs_and_prev_actions(
                    base, chosen_this_step, self._earlier_by_agent[agent]
                )
                x = x.unsqueeze(0)
                probs, value = self.policy_nets[i](x)
                dist = torch.distributions.Categorical(probs)
                a = dist.sample()
                actions[agent] = int(a.item())
                logps[agent] = dist.log_prob(a)
                values[agent] = value.squeeze(-1)
                chosen_this_step[agent] = actions[agent]

        return actions, logps, values

    # ---------- returns/advantage helpers ----------
    def _compute_advantages(self, rewards, values, next_value, dones_mask):
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        R = next_value.detach().view(1)
        R = R * (1 - dones_mask[-1])
        for t in reversed(range(T)):
            R = rewards[t] + self.args.gamma * R * (1 - dones_mask[t])
            returns[t] = R
        advantages = returns - values.squeeze(-1)
        return returns, advantages

    def _compute_GAE(self, rewards, values, next_value, dones_mask):
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=rewards.device)
        next_value = next_value.view(1, 1).detach()
        vals = torch.cat([values, next_value], dim=0)
        for t in reversed(range(T)):
            delta = rewards[t] + self.args.gamma * vals[t+1] * (1 - dones_mask[t]) - vals[t]
            gae = delta + self.args.gamma * self.args.lambda_GAE * (1 - dones_mask[t]) * gae
            advantages[t] = gae
        returns = advantages + vals[:-1].squeeze(-1)
        return returns, advantages

    @torch.no_grad()
    def _to_device(self, x: np.ndarray) -> torch.Tensor:
        """
        Flatten obs -> torch.Tensor on self.device.
        """
        t = torch.tensor(np.asarray(x, dtype=np.float32)).flatten()
        return t.to(self.device)

    @torch.no_grad()
    def policy_distribution(self, obs_dict: dict) -> dict:
        """
        Compute each agent's action probability distribution for the CURRENT obs_dict
        without doing any learning.

        Uses self.turn_groups to know move order (Nash vs Stackelberg).
        For each group in turn_groups:
            - Agents in earlier groups are considered 'leaders'
            - Agents in later groups observe one-hots of those earlier actions.
        We always use argmax actions for conditioning when building later agents'
        inputs, because that's what followers would expect in Stackelberg play.

        Returns:
            probs_out = {agent_id: probs_np} where probs_np is shape [n_actions]
        """
        probs_out = {}
        chosen_actions_turn = {}  # {agent_id: int_action} for THIS timestep

        # Flatten "all earlier groups" progressively
        earlier_agents_so_far: list[str] = []

        # We'll need a map from agent_id to its policy_net index
        # assuming self.agent_order and self.policy_nets are aligned 1:1
        index_of = {ag: i for i, ag in enumerate(self.agent_order)}

        for grp in self.turn_groups:
            # grp is a list[str] of agents that act at the same stage
            for ag in grp:
                # build model input for this agent
                obs_vec = self._to_device(obs_dict[ag])

                x = self._stackelberg_input_from_obs_and_prev_actions(
                    obs_vec=obs_vec,
                    prev_actions=chosen_actions_turn,
                    earlier_list=earlier_agents_so_far,
                )

                # forward pass
                net_idx = index_of[ag]
                self.policy_nets[net_idx].eval()
                probs, _ = self.policy_nets[net_idx](x)

                if probs.dim() > 1:
                    probs = probs.squeeze(0)
                probs_np = probs.cpu().numpy()

                probs_out[ag] = probs_np

                # store argmax action for conditioning of later groups
                greedy_act = int(np.argmax(probs_np))
                chosen_actions_turn[ag] = greedy_act

            # after this group acts, they become "earlier" for all future groups
            earlier_agents_so_far.extend(grp)

        return probs_out

    @torch.no_grad()
    def rollout_episode(self, greedy: bool = True, max_steps: int | None = None):
        """
        Run ONE full episode using frozen policies, no gradient, no buffer writes.

        greedy:
            True  -> choose argmax(action_probs)
            False -> multinomial sample from action_probs
        max_steps:
            upper bound on env steps (defaults to self.args.max_steps)

        Returns:
            traj = {
                "states": list[ dict(agent_id -> obs_at_t) ],
                "actions": list[ dict(agent_id -> int_action_at_t) ],
                "rewards": list[ dict(agent_id -> reward_at_t) ],
                "done_step": int,
                "episode_return": {agent_id: float_total_return}
            }
        """
        if max_steps is None:
            max_steps = getattr(self.args, "max_steps", 1000)

        # reset env
        obs, _ = self.env.reset()

        traj_states = []
        traj_actions = []
        traj_rewards = []
        ep_return = {ag: 0.0 for ag in self.agent_order}

        # helper for policy forward (same logic as policy_distribution but returns sampled action)
        def decide_actions_for_obs(curr_obs: dict) -> dict:
            chosen_actions_turn = {}
            earlier_agents_so_far: list[str] = []
            actions_for_env = {}

            index_of = {ag: i for i, ag in enumerate(self.agent_order)}

            for grp in self.turn_groups:
                for ag in grp:
                    obs_vec = self._to_device(curr_obs[ag])

                    x = self._stackelberg_input_from_obs_and_prev_actions(
                        obs_vec=obs_vec,
                        prev_actions=chosen_actions_turn,
                        earlier_list=earlier_agents_so_far,
                    )

                    net_idx = index_of[ag]
                    self.policy_nets[net_idx].eval()
                    probs, _ = self.policy_nets[net_idx](x)
                    if probs.dim() > 1:
                        probs = probs.squeeze(0)

                    probs_np = probs.cpu().numpy()

                    if greedy:
                        act = int(np.argmax(probs_np))
                    else:
                        act = int(np.random.choice(len(probs_np), p=probs_np / probs_np.sum()))

                    chosen_actions_turn[ag] = act
                    actions_for_env[ag] = act

                earlier_agents_so_far.extend(grp)

            return actions_for_env

        # rollout loop
        for t in range(max_steps):
            traj_states.append({ag: np.copy(obs[ag]) for ag in self.agent_order})

            step_actions = decide_actions_for_obs(obs)
            traj_actions.append(step_actions)

            next_obs, rew, terminated, truncated, info = self.env.step(step_actions)

            traj_rewards.append(rew)
            for ag in self.agent_order:
                ep_return[ag] += float(rew.get(ag, 0.0))

            done = False
            if any(terminated.values()) or any(truncated.values()):
                done = True

            obs = next_obs
            if done:
                done_step = t
                break
        else:
            done_step = max_steps - 1  # hit safety cap

        traj = {
            "states": traj_states,
            "actions": traj_actions,
            "rewards": traj_rewards,
            "done_step": done_step,
            "episode_return": ep_return,
        }
        return traj


    # ---------- PPO update (minibatch, per-agent) ----------
    def update(self, global_step: int, do_print):
        A = self.args
        dones_mask = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)

        actor_loss_list, critic_loss_list, entropy_list = [], [], []
        returns_per_agent, adv_per_agent, adv_norm_cache = {}, {}, {}

        # ----- Precompute returns/advantages per agent (Stackelberg-aware bootstrap) -----
        for i, agent in enumerate(self.agent_order):
            old_states       = torch.stack(self.buffer.states[agent]).to(self.device)
            old_actions      = torch.stack(self.buffer.actions[agent]).to(self.device)
            old_rewards      = torch.stack(self.buffer.rewards[agent]).to(self.device)
            old_next_states  = torch.stack(self.buffer.next_states[agent]).to(self.device)
            old_log_probs    = torch.stack(self.buffer.logprobs[agent]).to(self.device)
            old_values       = torch.stack(self.buffer.values[agent]).to(self.device)

            with torch.no_grad():
                xnv  = old_next_states[-1]
                prev = {}
                for ej in self._earlier_by_agent[agent]:
                    prev[ej] = int(torch.stack(self.buffer.actions[ej])[-1].item())
                xnv = self._stackelberg_input_from_obs_and_prev_actions(
                    xnv, prev, self._earlier_by_agent[agent]
                )
                xnv = xnv.unsqueeze(0)  # policy expects batch
                nv  = self.old_policy_nets[i](xnv)[1].squeeze(-1)

            if A.is_GAE:
                rets, advs = self._compute_GAE(old_rewards, old_values.view(-1, 1), nv, dones_mask)
            else:
                rets, advs = self._compute_advantages(old_rewards, old_values, nv, dones_mask)

            advs = advs.to(self.device)
            adv_norm = (advs - advs.mean()) / advs.std().clamp_min(1e-8)

            returns_per_agent[agent] = rets.to(self.device)
            adv_per_agent[agent]     = advs
            adv_norm_cache[agent]    = adv_norm

        # ----- PPO epochs with minibatches (Stackelberg-aware state augmentation) -----
        for epoch in range(A.num_epochs):
            # accumulate metrics across all agents and minibatches in this epoch
            epoch_actor_losses, epoch_critic_losses, epoch_entropies = [], [], []

            for i, agent in enumerate(self.agent_order):
                old_states      = torch.stack(self.buffer.states[agent]).to(self.device)
                old_actions     = torch.stack(self.buffer.actions[agent]).to(self.device)
                old_log_probs   = torch.stack(self.buffer.logprobs[agent]).to(self.device)
                target_returns  = returns_per_agent[agent]
                advantages      = adv_norm_cache[agent]

                # full-sequence augmentation once; slice per minibatch
                states_aug_full = self._augment_states(agent, old_states)

                N  = old_states.shape[0]
                idx = torch.randperm(N, device=self.device)
                mb  = A.minibatch_size if A.minibatch_size > 0 else N

                agent_policy_losses, agent_value_losses, agent_entropies = [], [], []

                for start in range(0, N, mb):
                    mb_idx       = idx[start:start+mb]
                    mb_states    = states_aug_full[mb_idx]
                    mb_actions   = old_actions[mb_idx]
                    mb_log_probs = old_log_probs[mb_idx]
                    mb_returns   = target_returns[mb_idx]
                    mb_advs      = advantages[mb_idx]

                    opt = self.optimizers[i]
                    opt.zero_grad(set_to_none=True)

                    probs, values = self.policy_nets[i](mb_states)
                    dist = torch.distributions.Categorical(probs)

                    # policy loss
                    new_log_probs = dist.log_prob(mb_actions)
                    ratios = torch.exp(new_log_probs - mb_log_probs.detach())
                    surr1 = ratios * mb_advs
                    surr2 = torch.clamp(ratios, 1 - A.epsilon, 1 + A.epsilon) * mb_advs
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value loss (raw MSE, before weighting)
                    value_pred = values.squeeze(-1)
                    vf_loss_unscaled = self.critic_loss(value_pred, mb_returns)

                    # entropy bonus
                    entropy = dist.entropy().mean()

                    # total loss with value_coef and entropy_coef
                    total_loss = policy_loss \
                              + A.value_coef * vf_loss_unscaled \
                              - A.entropy_coef * entropy

                    total_loss.backward()

                    # global grad clip across all params of this net
                    if getattr(A, "grad_clip", None) is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy_nets[i].parameters(),
                            A.grad_clip
                        )

                    opt.step()

                    # logging (store these, then average later)
                    agent_policy_losses.append(policy_loss.item())
                    agent_value_losses.append(vf_loss_unscaled.item())
                    agent_entropies.append(entropy.item())

                # ===== Log per-agent average per epoch =====
                mean_actor   = float(np.mean(agent_policy_losses)) if agent_policy_losses else 0.0
                mean_critic  = float(np.mean(agent_value_losses)) if agent_value_losses else 0.0
                mean_entropy = float(np.mean(agent_entropies)) if agent_entropies else 0.0
                dom_ratio    = mean_critic / (abs(mean_actor) + 1e-8)

                self.writer.add_scalar(f"{agent}/actor_term", mean_actor, global_step)
                self.writer.add_scalar(f"{agent}/value_term", mean_critic, global_step)
                self.writer.add_scalar(f"{agent}/entropy_term", -A.entropy_coef * mean_entropy, global_step)
                self.writer.add_scalar(f"{agent}/dom_ratio_v_over_a", dom_ratio, global_step)

                epoch_actor_losses.append(mean_actor)
                epoch_critic_losses.append(mean_critic)
                epoch_entropies.append(mean_entropy)

        # ===== Sync target/old nets used for bootstrap =====
        for i in range(len(self.policy_nets)):
            self.old_policy_nets[i].load_state_dict(self.policy_nets[i].state_dict())

        # ===== Log overall averages per update =====
        mean_actor_all   = float(np.mean(epoch_actor_losses)) if epoch_actor_losses else 0.0
        mean_critic_all  = float(np.mean(epoch_critic_losses)) if epoch_critic_losses else 0.0
        mean_entropy_all = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0

        self.writer.add_scalar("loss/actor_mean", mean_actor_all, global_step)
        self.writer.add_scalar("loss/critic_mean", mean_critic_all, global_step)
        self.writer.add_scalar("loss/entropy_mean", mean_entropy_all, global_step)

        self.history["loss_history"].append({
            "step": int(global_step),
            "actor": mean_actor_all,
            "critic": mean_critic_all,
            "entropy": mean_entropy_all
        })

        return mean_actor_all, mean_critic_all, mean_entropy_all



    # -------- training loop --------
    def train(self):
        A = self.args
        episode_rewards = []  # list of dicts {agent: ep_return}
        ep_ret = {a: 0.0 for a in self.agent_order}
        obs = self._last_obs
        steps = 0
        start_time = datetime.now().replace(microsecond=0)

        while steps < A.total_timesteps:
            steps += 1

            actions, logps, values = self._choose_actions(obs)
            next_obs, rewards, term, trunc, infos = self.env.step(actions)
            done = all((term.get(a, False) or trunc.get(a, False)) for a in self.agent_order)

            for i, agent in enumerate(self.agent_order):
                s  = obs.get(agent,  np.zeros(self.env.observation_space(agent).shape, dtype=np.float32))
                ns = next_obs.get(agent, np.zeros_like(s))
                s_t  = self._obs_to_tensor(s, agent)
                ns_t = self._obs_to_tensor(ns, agent)
                self.buffer.states[agent].append(s_t)
                self.buffer.next_states[agent].append(ns_t)

                if agent in actions:
                    a_t = torch.tensor(actions[agent], dtype=torch.long, device=self.device)
                    lp_t = logps[agent].to(self.device)
                    v_t  = values[agent].to(self.device)
                else:
                    a_t = torch.tensor(0, dtype=torch.long, device=self.device)
                    lp_t = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    v_t  = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                self.buffer.actions[agent].append(a_t)
                self.buffer.logprobs[agent].append(lp_t)
                self.buffer.values[agent].append(v_t)

                r = float(rewards.get(agent, 0.0))
                self.buffer.rewards[agent].append(torch.tensor(r, dtype=torch.float32, device=self.device))
                ep_ret[agent] += r

            self.buffer.dones.append(torch.tensor(float(done), dtype=torch.float32, device=self.device))
            obs = next_obs

            if done:
                # per-episode history + TB logs
                episode_rewards.append(ep_ret.copy())
                self.history["episode_returns"].append(ep_ret.copy())
                for agent in self.agent_order:
                    self.writer.add_scalar(f"{agent}/episode_return", ep_ret[agent], steps)
                ep_ret = {a: 0.0 for a in self.agent_order}
                reset_out = self.env.reset(seed=None)
                obs, _ = (reset_out if isinstance(reset_out, tuple) else (reset_out, {}))

            # periodic PPO update
            if steps % A.num_steps == 0:
                do_print = True
                mean_actor, mean_critic, mean_entropy = self.update(global_step=steps, do_print=do_print)
                self.buffer.clear()
                self.old_policy_nets = [copy.deepcopy(net) for net in self.policy_nets]

                # moving mean of last 10 episodes (if any)
                if len(episode_rewards) >= 1:
                    last10 = episode_rewards[-10:]
                    for agent in self.agent_order:
                        mean10 = float(np.mean([er[agent] for er in last10]))
                        self.writer.add_scalar(f"{agent}/SmoothReturn10", mean10, steps)

            if steps % A.log_steps == 0 and episode_rewards:
                last = episode_rewards[-1]
                elapsed = datetime.now().replace(microsecond=0) - start_time
                print(f"[Steps: {steps}] Return: {last} | Elapsed: {elapsed}")

        self.env.close()
        self.writer.close()

        # ---- Save training history to disk ----
        self._save_history()

        return episode_rewards

    def _save_history(self):
        """Save episode returns and loss history to {save_path}.npz and {save_path}.json"""
        path = Path(self.args.save_path)
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")

        # Pack arrays for npz (ragged dicts -> json string fallback)
        ep_json = json.dumps(self.history["episode_returns"])
        loss_json = json.dumps(self.history["loss_history"])
        np.savez_compressed(npz_path, episode_returns_json=ep_json, loss_history_json=loss_json)

        # Also save human-readable JSON
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"📦 Saved training history:\n - {npz_path}\n - {json_path}")
