import numpy as np
import torch
import torch.nn.functional as F

from agent import Agent
from agent.utils import soft_update
from agent.model import QNetwork, GaussianPolicy, QEnsemble


class CALAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda")
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.args = args

        self.update_counter = 0

        # Safety related params
        self.c = args.c
        self.cost_lr_scale = 1.

        # Reward critic
        if args.rew_ens:
            self.critic = QEnsemble(num_inputs, action_space.shape[0], args.qr_ens_size, args.hidden_size).to(device=self.device)
            self.critic_target = QEnsemble(num_inputs, action_space.shape[0], args.qr_ens_size, args.hidden_size).to(self.device)
        else:
            self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Safety critics
        self.safety_critics = QEnsemble(num_inputs, action_space.shape[0], args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets = QEnsemble(num_inputs, action_space.shape[0], args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())

        # policy
        self.policy = GaussianPolicy(args, num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

        self.log_lam = torch.tensor(np.log(np.clip(0.6931, 1e-8, 1e8))).to(self.device)
        self.log_lam.requires_grad = True

        self.kappa = 0

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=args.qr_lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=args.qc_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=args.lr)

        self.train()
        self.critic_target.train()
        self.safety_critic_targets.train()

        # Set target cost
        if args.safetygym:
            self.target_cost = args.cost_lim * (1 - self.safety_discount**args.epoch_length) / (
                1 - self.safety_discount) / args.epoch_length if self.safety_discount < 1 else args.cost_lim
        else:
            self.target_cost = args.cost_lim
        print("Constraint Budget: ", self.target_cost)


    def train(self, training=True):
        self.training = training
        self.policy.train(training)
        self.critic.train(training)
        self.safety_critics.train(training)


    @property
    def alpha(self):
        return self.log_alpha.exp()


    @property
    def lam(self):
        return self.log_lam.exp()


    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    def update_critic(self, state, action, reward, cost, next_state, mask):
        next_action, next_log_prob, _ = self.policy.sample(next_state)

        # Reward critics update
        if self.args.rew_ens:
            qr_idxs = np.random.choice(self.args.qr_ens_size, 2)
            current_QRs = self.critic(state, action)
            with torch.no_grad():
                target_QRs = self.critic_target(next_state, next_action)
                target_V = target_QRs[qr_idxs].min(dim=0).values - self.alpha.detach() * next_log_prob
        else:
            current_Q1, current_Q2 = self.critic(state, action)
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
        target_Q = reward + (mask * self.discount * target_V)
        target_Q = target_Q.detach()

        if self.args.rew_ens:
            critic_loss = F.mse_loss(current_QRs, target_Q[None, :, :].repeat(self.args.qr_ens_size, 1, 1))
        else:
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Safety critics update
        qc_idxs = np.random.choice(self.args.qc_ens_size, self.args.M)
        current_QCs = self.safety_critics(state, action) # shape(E, B, 1)
        with torch.no_grad():
            next_QCs = self.safety_critic_targets(next_state, next_action)
        next_QC_random_max = next_QCs[qc_idxs].max(dim=0, keepdim=True).values

        if self.args.safetygym:
            mask = torch.ones_like(mask).to(self.device)
        next_QC = next_QC_random_max.repeat(self.args.qc_ens_size, 1, 1) if self.args.intrgt_max else next_QCs
        target_QCs = cost[None, :, :].repeat(self.args.qc_ens_size, 1, 1) + \
                    (mask[None, :, :].repeat(self.args.qc_ens_size, 1, 1) * self.safety_discount * next_QC)
        safety_critic_loss = F.mse_loss(current_QCs, target_QCs.detach())

        self.safety_critic_optimizer.zero_grad()
        safety_critic_loss.backward()
        self.safety_critic_optimizer.step()


    def update_actor(self, state, action_taken, init_state=None):
        action, log_prob, _ = self.policy.sample(state)

        # Reward critic
        if self.args.rew_ens:
            actor_QRs = self.critic(state, action)
            qr_idxs = np.random.choice(self.args.qr_ens_size, 2)
            if self.args.qr_mean:
                actor_Q = torch.mean(actor_QRs, dim=0)
            else:
                actor_Q = torch.min(actor_QRs[qr_idxs], dim=0).values
        else:
            actor_Q1, actor_Q2 = self.critic(state, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)

        # Safety critic
        # with torch.no_grad():
        #     current_QCs = self.safety_critics(state, action_taken)
        #     current_std, current_mean = torch.std_mean(current_QCs, dim=0)
        #     if self.args.qc_ens_size == 1:
        #         current_std = torch.zeros_like(current_mean).to(self.device)
        #     current_QC = current_mean + self.args.k * current_std
        actor_QCs = self.safety_critics(state, action)
        actor_std, actor_mean = torch.std_mean(actor_QCs, dim=0)
        if self.args.qc_ens_size == 1:
            actor_std = torch.zeros_like(actor_std).to(self.device)
        actor_QC = actor_mean + self.args.k * actor_std

        # Compute gradient rectification
        # self.rect = self.c * torch.mean(self.target_cost - current_QC)
        self.rect = self.c * torch.mean(self.target_cost - actor_mean.detach()) # TODO 先不用init_state, 看这里的变化有什么影响
        self.rect = torch.clamp(self.rect.detach(), max=self.lam.item())

        # Policy loss
        lam = self.lam.detach()
        if self.args.woALM:
            actor_loss = torch.mean(
                self.alpha.detach() * log_prob
                - actor_Q
                + (lam) * actor_QC
            )
        else:
            actor_loss = torch.mean(
                self.alpha.detach() * log_prob
                - actor_Q
                + (lam - self.rect) * actor_QC
            )

        # Optimize the policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = torch.mean(self.alpha * (-log_prob - self.target_entropy).detach())
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Dual update
        if init_state is not None:
            with torch.no_grad():
                current_action, _, _ = self.policy.sample(state)
                current_QCs = self.safety_critics(init_state, current_action)
                current_QC = torch.mean(current_QCs, dim=0)
        else:
            current_QC = actor_QC.detach()
        self.log_lam_optimizer.zero_grad()
        lam_loss = torch.mean(self.lam * (self.target_cost - current_QC).detach())
        lam_loss.backward()
        self.log_lam_optimizer.step()


    def update_parameters(self, memory, updates):
        self.update_counter += 1
        if self.args.init_s_for_qc:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, init_s_batch = memory
            init_s_batch = torch.FloatTensor(init_s_batch).to(self.device)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        cost_batch = torch.FloatTensor(reward_batch[:, 1]).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch[:, 0]).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        self.update_critic(state_batch, action_batch, reward_batch, cost_batch, next_state_batch, mask_batch)
        if self.args.init_s_for_qc:
            self.update_actor(state_batch, action_batch, init_s_batch)
        else:
            self.update_actor(state_batch, action_batch)

        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)
            soft_update(self.safety_critic_targets, self.safety_critics, self.critic_tau)
