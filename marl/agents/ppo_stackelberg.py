import torch, copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from marl.agents.ppo_base import NashPPO
from marl.nets import ActorCriticPPO


class StackelbergPPO(NashPPO):
    """
    Hierarchical Stackelberg PPO agent for PettingZoo Parallel envs.
    Groups act in order; each agent sees the state plus one-hots of all earlier groups' actions.
    """

    def __init__(self, env, args, action_sequence=None):
        super().__init__(env, args)

        if action_sequence is None:
            self.turn_groups = [[ag] for ag in self.agent_order]
        else:
            self.turn_groups = action_sequence
            self.agent_order = [a for grp in self.turn_groups for a in grp]

        self.flat_order = [a for grp in self.turn_groups for a in grp]
        self._index_of_agent = {ag: i for i, ag in enumerate(self.flat_order)}

        self._earlier_by_agent = {}
        seen = []
        for grp in self.turn_groups:
            for ag in grp:
                self._earlier_by_agent[ag] = list(seen)
            seen.extend(grp)

        self.policy_nets = []
        for ag in self.flat_order:
            obs_dim = int(np.prod(self.env.observation_space(ag).shape))
            extra = sum(self.env.action_space(ea).n for ea in self._earlier_by_agent[ag])
            n_actions = self.env.action_space(ag).n
            self.policy_nets.append(ActorCriticPPO(obs_dim + extra, n_actions).to(self.device))

        self.old_policy_nets = [copy.deepcopy(n) for n in self.policy_nets]
        self.optimizers = [torch.optim.Adam(n.parameters(), lr=self.args.lr) for n in self.policy_nets]
        self.writer = SummaryWriter(log_dir="runs/ppo_stackelberg_pz")

    def _onehot(self, agent_id, a_tensor):
        n = self.env.action_space(agent_id).n
        return torch.nn.functional.one_hot(a_tensor.long(), num_classes=n).float().to(self.device)

    def _stackelberg_input_from_obs_and_prev(self, obs_vec, chosen_actions, earlier_agents):
        x = obs_vec
        for aj in earlier_agents:
            if aj in chosen_actions:
                oh = self._onehot(aj, torch.tensor(chosen_actions[aj], device=self.device))
            else:
                oh = torch.zeros(self.env.action_space(aj).n, device=self.device)
            x = torch.cat([x, oh], dim=-1)
        return x

    def _choose_actions(self, obs_dict):
        acts, logps, vals = {}, {}, {}
        chosen = {}
        for grp in self.turn_groups:
            for ag in grp:
                if ag not in self.env.agents:
                    continue
                with torch.no_grad():
                    base = self._obs_to_tensor(obs_dict[ag], ag)
                    earlier = self._earlier_by_agent[ag]
                    x = self._stackelberg_input_from_obs_and_prev(base, chosen, earlier)
                    i = self._index_of_agent[ag]
                    probs, v = self.policy_nets[i](x)
                    dist = torch.distributions.Categorical(probs)
                    a = dist.sample()
                    acts[ag] = int(a.item())
                    logps[ag] = dist.log_prob(a)
                    vals[ag] = v.squeeze(-1)
            for ag in grp:
                if ag in acts:
                    chosen[ag] = acts[ag]
        return acts, logps, vals

    def update(self, global_step):
        import time
        t_up0 = time.perf_counter()

        dones_mask = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        actor_loss_list, critic_loss_list, entropy_list = [], [], []
        returns_per_agent, adv_per_agent = {}, {}

        t_fwd0 = time.perf_counter()
        for i, agent in enumerate(self.agent_order):
            old_states = torch.stack(self.buffer.states[agent]).to(self.device)
            old_actions = torch.stack(self.buffer.actions[agent]).to(self.device)
            old_rewards = torch.stack(self.buffer.rewards[agent]).to(self.device)
            old_next_states = torch.stack(self.buffer.next_states[agent]).to(self.device)
            old_log_probs = torch.stack(self.buffer.logprobs[agent]).to(self.device)
            old_values = torch.stack(self.buffer.values[agent]).to(self.device)

            with torch.no_grad():
                xnv = old_next_states[-1]
                prev = {}
                for ej in self._earlier_by_agent[agent]:
                    prev[ej] = int(torch.stack(self.buffer.actions[ej])[-1].item())
                xnv = self._stackelberg_input_from_obs_and_prev(xnv, prev, self._earlier_by_agent[agent])
                nv = self.old_policy_nets[i](xnv)[1].squeeze(-1)

            if self.args.is_GAE:
                rets, advs = self._compute_GAE(old_rewards, old_values.view(-1, 1), nv, dones_mask)
            else:
                rets, advs = self._compute_advantages(old_rewards, old_values, nv, dones_mask)
            returns_per_agent[agent] = rets
            adv_per_agent[agent] = advs
        self.timing["ppo_forward_time"] += time.perf_counter() - t_fwd0

        # ----- PPO epochs -----
        for _ in range(self.args.num_epochs):
            t_fwd = time.perf_counter()
            batches = []
            for i, agent in enumerate(self.agent_order):
                s = torch.stack(self.buffer.states[agent]).to(self.device)              # [T, obs]
                a = torch.stack(self.buffer.actions[agent]).to(self.device)            # [T]
                lp_old = torch.stack(self.buffer.logprobs[agent]).to(self.device)      # [T]
                ret = returns_per_agent[agent].to(self.device)                         # [T]
                adv = adv_per_agent[agent].to(self.device)                             # [T]

                # Augment states with earlier agents' actions at each timestep
                s_aug = self._augment_states(agent, s)                                 # [T, obs+sum(onehots)]

                probs, values = self.policy_nets[i](s_aug)
                dist = torch.distributions.Categorical(probs)
                lp_new = dist.log_prob(a)
                ratios = torch.exp(lp_new - lp_old.detach())

                s1 = ratios * adv
                s2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv
                actor_loss = -torch.min(s1, s2).mean()

                critic_loss = self.critic_loss(ret, values.squeeze(-1))
                entropy = dist.entropy().mean()

                loss = actor_loss + self.args.value_coef * critic_loss - self.args.entropy_coef * entropy
                batches.append((i, loss, actor_loss, critic_loss, entropy))
            self.timing["ppo_forward_time"] += time.perf_counter() - t_fwd

            t_bwd = time.perf_counter()
            for i, loss, al, cl, ent in batches:
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()
                actor_loss_list.append(al.item())
                critic_loss_list.append(cl.item())
                entropy_list.append(ent.item())
            self.timing["ppo_backward_time"] += time.perf_counter() - t_bwd

        for i in range(len(self.policy_nets)):
            self.old_policy_nets[i].load_state_dict(self.policy_nets[i].state_dict())

        self.timing["ppo_update_time"] += time.perf_counter() - t_up0
        return np.mean(actor_loss_list), np.mean(critic_loss_list), np.mean(entropy_list)
           
