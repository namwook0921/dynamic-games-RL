import torch, copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from marl.agents.ppo_base import NashPPO
from marl.nets import ActorCriticPPO



class StackelbergPPO(NashPPO):
    """
    Hierarchical Stackelberg PPO agent for PettingZoo Parallel envs.
    Leader–follower order follows self.agent_order.
    """

    def __init__(self, env, args):
        super().__init__(env, args)
        # rebuild policy nets to include one-hot of earlier agents' actions
        self.policy_nets = []
        for i, ag in enumerate(self.agent_order):
            obs_dim = int(np.prod(self.env.observation_space(ag).shape))
            extra = sum(self.env.action_space(a).n for a in self.agent_order[:i])
            n_actions = self.env.action_space(ag).n
            self.policy_nets.append(ActorCriticPPO(obs_dim+extra, n_actions).to(self.device))
        self.old_policy_nets=[copy.deepcopy(n) for n in self.policy_nets]
        self.optimizers=[torch.optim.Adam(n.parameters(), lr=self.args.lr) for n in self.policy_nets]
        self.writer = SummaryWriter(log_dir="runs/ppo_stackelberg_pz")

    # ---------- helper ----------
    def _onehot(self, idx, a_tensor):
        n=self.env.action_space(self.agent_order[idx]).n
        return torch.nn.functional.one_hot(a_tensor.long(), num_classes=n).float().to(self.device)

    def _stackelberg_input(self, obs_vec, chosen, upto_i):
        x=obs_vec
        for j in range(upto_i):
            aj=self.agent_order[j]
            if aj in chosen:
                oh=self._onehot(j, torch.tensor(chosen[aj], device=self.device))
            else:
                oh=torch.zeros(self.env.action_space(aj).n, device=self.device)
            x=torch.cat([x,oh],dim=-1)
        return x

    # ---------- override choose ----------
    def _choose_actions(self, obs_dict):
        acts, logps, vals={}, {}, {}
        chosen={}
        for i,ag in enumerate(self.agent_order):
            if ag not in self.env.agents: continue
            with torch.no_grad():
                base=self._obs_to_tensor(obs_dict[ag], ag)
                x=self._stackelberg_input(base, chosen, i)
                probs,v=self.policy_nets[i](x)
                dist=torch.distributions.Categorical(probs)
                a=dist.sample()
                acts[ag]=int(a.item()); logps[ag]=dist.log_prob(a); vals[ag]=v.squeeze(-1)
                chosen[ag]=acts[ag]
        return acts, logps, vals

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
           
