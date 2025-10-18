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
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        actor_loss, critic_loss, entropy = [], [], []
        returns, advs = {}, {}

        for ag in self.flat_order:
            i = self._index_of_agent[ag]
            s = torch.stack(self.buffer.states[ag]).to(self.device)
            a = torch.stack(self.buffer.actions[ag]).to(self.device)
            r = torch.stack(self.buffer.rewards[ag]).to(self.device)
            ns = torch.stack(self.buffer.next_states[ag]).to(self.device)
            v = torch.stack(self.buffer.values[ag]).to(self.device)

            with torch.no_grad():
                xnv = ns[-1]
                prev = {}
                for ej in self._earlier_by_agent[ag]:
                    prev[ej] = int(torch.stack(self.buffer.actions[ej])[-1].item())
                xnv = self._stackelberg_input_from_obs_and_prev(xnv, prev, self._earlier_by_agent[ag])
                nv = self.old_policy_nets[i](xnv)[1].squeeze(-1)

            if self.args.is_GAE:
                rets, ad = self._compute_GAE(r, v.view(-1, 1), nv, dones)
            else:
                rets, ad = self._compute_advantages(r, v, nv, dones)
            returns[ag], advs[ag] = rets, ad

    # ---------- override update ----------
    def update(self, global_step):
        # identical to base but augments obs with earlier one-hots
        dones=torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        actor_loss, critic_loss, entropy = [],[],[]
        returns, advs = {}, {}
        for i,ag in enumerate(self.agent_order):
            s=torch.stack(self.buffer.states[ag]).to(self.device)
            a=torch.stack(self.buffer.actions[ag]).to(self.device)
            r=torch.stack(self.buffer.rewards[ag]).to(self.device)
            ns=torch.stack(self.buffer.next_states[ag]).to(self.device)
            v=torch.stack(self.buffer.values[ag]).to(self.device)
            with torch.no_grad():
                xnv=ns[-1]; prev={}
                for j in range(i):
                    aj=self.agent_order[j]
                    prev[aj]=int(torch.stack(self.buffer.actions[aj])[-1].item())
                xnv=self._stackelberg_input(xnv, prev, i)
                nv=self.old_policy_nets[i](xnv)[1].squeeze(-1)
            rets,ad=(self._compute_GAE(r,v.view(-1,1),nv,dones)
                     if self.args.is_GAE else
                     self._compute_advantages(r,v,nv,dones))
            returns[ag],advs[ag]=rets,ad
        for _ in range(self.args.num_epochs):
            for ag in self.flat_order:
                i = self._index_of_agent[ag]
                n_steps = len(self.buffer.states[ag])
                idx = np.arange(n_steps)
                np.random.shuffle(idx)

                for start in range(0, n_steps, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_idx = idx[start:end]

                    s = torch.stack(self.buffer.states[ag])[mb_idx].to(self.device)
                    a = torch.stack(self.buffer.actions[ag])[mb_idx].to(self.device)
                    old_logp = torch.stack(self.buffer.logprobs[ag])[mb_idx].to(self.device)
                    adv = advs[ag][mb_idx].detach()
                    ret = returns[ag][mb_idx].detach()

                    chosen_prev = {}
                    for ej in self._earlier_by_agent[ag]:
                        chosen_prev[ej] = int(torch.stack(self.buffer.actions[ej])[-1].item())
                    x = self._stackelberg_input_from_obs_and_prev(s, chosen_prev, self._earlier_by_agent[ag])

                    probs, values = self.policy_nets[i](x)
                    dist = torch.distributions.Categorical(probs)
                    logp = dist.log_prob(a)
                    ratio = torch.exp(logp - old_logp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = 0.5 * (ret - values.squeeze(-1)).pow(2).mean()
                    entropy = dist.entropy().mean()

                    loss = (policy_loss +
                            self.args.value_coef * value_loss -
                            self.args.entropy_coef * entropy)

                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                    self.writer.add_scalar(f"{ag}/policy_loss", policy_loss.item(), global_step)
                    self.writer.add_scalar(f"{ag}/value_loss", value_loss.item(), global_step)
                    self.writer.add_scalar(f"{ag}/entropy", entropy.item(), global_step)
                    
        for i in range(len(self.policy_nets)):
            self.old_policy_nets[i].load_state_dict(self.policy_nets[i].state_dict())
           
