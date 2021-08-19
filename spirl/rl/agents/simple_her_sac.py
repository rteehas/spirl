import torch
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.components.agent import BaseAgent
from spirl.utils.general_utils import ParamDict, map_dict, AttrDict
from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np
from spirl.rl.utils.mpi import sync_networks
import copy


class HER_Agent(SACAgent):

    def __init__(self, config):
        SACAgent.__init__(self, config)
        self._hp = self._default_hparams().overwrite(config)
        self.policy = self._hp.policy(self._hp.policy_params)
        if self.policy.has_trainable_params:
            self.policy_opt = self._get_optimizer(self._hp.optimizer, self.policy, self._hp.policy_lr)

    def _default_hparams(self):
        default_dict = ParamDict({
            'critic': None,  # critic class
            'critic_params': None,  # parameters for the critic class
            'replay': None,  # replay buffer class
            'replay_params': None,  # parameters for replay buffer
            'critic_lr': 3e-4,  # learning rate for critic update
            'alpha_lr': 3e-4,  # learning rate for alpha coefficient update
            'reward_scale': 1.0,  # SAC reward scale
            'clip_q_target': False,  # if True, clips Q target
            'target_entropy': None,  # target value for automatic entropy tuning, if None uses -action_dim
            'regularization': 0.001,
            'her_iters': 100,
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        self.add_experience(experience_batch)

        T = self.replay_buffer.size

        for ep in self.her_iters:
            goal = self.sample_goal() # to impl
            for _ in range(T):
                filter = list(range(_))
                her_batch = self._sample_experience(filter)
                her_batch = self._normalize_batch(her_batch)
                her_batch = map2torch(her_batch, self._hp.device)
                her_batch = self._preprocess_experience(her_batch)

                rewards = self.get_gc_rewards(her_batch)
                obs_cat = self._concat(her_batch.observations, goal)
                goal_cat = self._concat(her_batch.next, goal)
                new_trans = AttrDict(observations=obs_cat, actions=her_batch.actions, rewards=rewards, next=goal_cat)
                self.add_experience(new_trans)

                goals = self.her_sample_experience(filter).actions

                batch = copy.deepcopy(her_batch)
                batch.next = goals

                rewards = self.get_gc_rewards(batch)
                obs_cat = self._concat(batch.observations, goals)
                goals_cat = self._concat(her_batch.next, goal)
                new_trans = AttrDict(observations=obs_cat, actions=batch.actions, rewards=rewards, next=goals_cat)
                self.add_experience(new_trans)


        for _ in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)

            policy_output = self._run_policy(experience_batch.observation)

            # update alpha
            alpha_loss = self._update_alpha(experience_batch, policy_output)

            # compute policy loss
            policy_loss = self._compute_policy_loss(experience_batch, policy_output)

            # compute target Q value
            with torch.no_grad():
                policy_output_next = self._run_policy(experience_batch.observation_next)
                value_next = self._compute_next_value(experience_batch, policy_output_next)
                q_target = experience_batch.reward * self._hp.reward_scale + \
                           (1 - experience_batch.done) * self._hp.discount_factor * value_next
                if self._hp.clip_q_target:
                    q_target = self._clip_q_target(q_target)
                q_target = q_target.detach()
                check_shape(q_target, [self._hp.batch_size])

            # compute critic loss
            critic_losses, qs = self._compute_critic_loss(experience_batch, q_target)

            # update critic networks
            [self._perform_update(critic_loss, critic_opt, critic)
             for critic_loss, critic_opt, critic in zip(critic_losses, self.critic_opts, self.critics)]

            # update target networks
            [self._soft_update_target_network(critic_target, critic)
             for critic_target, critic in zip(self.critic_targets, self.critics)]

            # update policy network on policy loss
            self._perform_update(policy_loss, self.policy_opt, self.policy)

            # logging
            info = AttrDict(  # losses
                policy_loss=policy_loss,
                alpha_loss=alpha_loss,
                critic_loss_1=critic_losses[0],
                critic_loss_2=critic_losses[1],
            )
            if self._update_steps % 100 == 0:
                info.update(AttrDict(  # gradient norms
                    policy_grad_norm=avg_grad_norm(self.policy),
                    critic_1_grad_norm=avg_grad_norm(self.critics[0]),
                    critic_2_grad_norm=avg_grad_norm(self.critics[1]),
                ))
            info.update(AttrDict(  # misc
                alpha=self.alpha,
                pi_log_prob=policy_output.log_prob.mean(),
                policy_entropy=policy_output.dist.entropy().mean(),
                q_target=q_target.mean(),
                q_1=qs[0].mean(),
                q_2=qs[1].mean(),
            ))
            info.update(self._aux_info(policy_output))
            info = map_dict(ten2ar, info)

        return info

    def her_sample_experience(self, filter):
        return self.replay_buffer.sample(n_samples=self._hp.batch_size, filter=filter)

    def _concat(self, obs, goal):
        return torch.cat([obs, goal], 1)

    def get_gc_rewards(self, batch):
        pass