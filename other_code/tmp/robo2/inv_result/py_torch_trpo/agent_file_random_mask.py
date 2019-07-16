import argparse
from itertools import count
import gym
import scipy.optimize
import roboschool
import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
import pickle 
from utils import *
import os, errno

torch.manual_seed(1)
np.random.seed(0)

torch.utils.backcompat.broadcast_warning.enabled = True

torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

class agent(object):
    def __init__(self, env, ar):
        self.env =  env
        self.args = ar
        self.num_inputs = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.policy_net = Policy(self.num_inputs, self.num_actions)
        self.value_net = Value(self.num_inputs)
        self.running_state = ZFilter((self.num_inputs,), clip=5)


    def learn(self, run_state_update = True):
        args = self.args
        num_inputs = self.num_inputs
        num_actions = self.num_actions
        env = self.env
        
        def select_action(state):
            state = torch.from_numpy(state).unsqueeze(0)
            action_mean, _, action_std = self.policy_net(Variable(state))
            action = torch.normal(action_mean, action_std)
            return action

        def update_params(batch):
            rewards = torch.Tensor(batch.reward)
            masks = torch.Tensor(batch.mask)
            actions = torch.Tensor(np.concatenate(batch.action, 0))
            states = torch.Tensor(batch.state)
            values = self.value_net(Variable(states))

            returns = torch.Tensor(actions.size(0),1)
            deltas = torch.Tensor(actions.size(0),1)
            advantages = torch.Tensor(actions.size(0),1)

            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            for i in reversed(range(rewards.size(0))):
                returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
                advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

                prev_return = returns[i, 0]
                prev_value = values.data[i, 0]
                prev_advantage = advantages[i, 0]

            targets = Variable(returns)

            # Original code uses the same LBFGS to optimize the value loss
            def get_value_loss(flat_params):
                set_flat_params_to(self.value_net, torch.Tensor(flat_params))
                for param in self.value_net.parameters():
                    if param.grad is not None:
                        param.grad.data.fill_(0)

                values_ = self.value_net(Variable(states))

                value_loss = (values_ - targets).pow(2).mean()

                # weight decay
                for param in self.value_net.parameters():
                    value_loss += param.pow(2).sum() * args.l2_reg
                value_loss.backward()
                return (value_loss.data.double().numpy()[0], get_flat_grad_from(self.value_net).data.double().numpy())

            flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(self.value_net).double().numpy(), maxiter=25)
            set_flat_params_to(self.value_net, torch.Tensor(flat_params))

            advantages = (advantages - advantages.mean()) / advantages.std()

            action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
            fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

            def get_loss(volatile=False):
                action_means, action_log_stds, action_stds = self.policy_net(Variable(states, volatile=volatile))
                log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
                return action_loss.mean()


            def get_kl():
                mean1, log_std1, std1 = self.policy_net(Variable(states))

                mean0 = Variable(mean1.data)
                log_std0 = Variable(log_std1.data)
                std0 = Variable(std1.data)
                kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                return kl.sum(1, keepdim=True)

            trpo_step(self.policy_net, get_loss, get_kl, args.max_kl, args.damping)

        
        running_reward = ZFilter((1,), demean=False, clip=10)

        for i_episode in range(args.max_epi):
            memory = Memory()

            num_steps = 0
            reward_batch = 0
            num_episodes = 0
            while num_steps < args.batch_size:
                state = env.reset()
                state = self.running_state(state, update=run_state_update)

                reward_sum = 0
                for t in range(10000): # Don't infinite loop while learning
                    action = select_action(state)
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = env.step(action)
                    reward_sum += reward

                    next_state = self.running_state(next_state, update=run_state_update)

                    mask = 1
                    if done:
                        mask = 0

                    memory.push(state, np.array([action]), mask, next_state, reward)

                    if done:
                        break

                    state = next_state
                num_steps += (t-1)
                num_episodes += 1
                reward_batch += reward_sum

            reward_batch /= num_episodes
            batch = memory.sample()
            
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                    i_episode, reward_sum, reward_batch))
            
            if args.max_avg<reward_batch:
                break
            
            update_params(batch)

            
            
            

        
    def select_action_deterministic(self,state):
        state = self.running_state(state, update=False)
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.policy_net(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action_mean.data[0].numpy()
    
    def select_action_stochastic(self,state):
        state = self.running_state(state, update=False)
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.policy_net(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action.data[0].numpy()
    
    def save_model(self,folder):
        
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        filehandler = open(folder+'/policy_net.pkl', 'wb') 
        pickle.dump(self.policy_net, filehandler) 
        filehandler.close()
        filehandler = open(folder+'/value_net.pkl', 'wb') 
        pickle.dump(self.value_net, filehandler) 
        filehandler.close()
        filehandler = open(folder+'/running.pkl', 'wb') 
        pickle.dump(self.running_state, filehandler) 
        filehandler.close()
        
    def load_model(self,folder):
        polic_n = pickle.load(open(folder+'/policy_net.pkl', 'rb'))
        value_n = pickle.load(open(folder+'/value_net.pkl', 'rb'))
        run_s = pickle.load(open(folder+'/running.pkl', 'rb'))
        self.policy_net = polic_n
        self.value_net = value_n
        self.running_state = run_s
    