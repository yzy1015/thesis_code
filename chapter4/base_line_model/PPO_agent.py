from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from baselines.ppo1 import mlp_policy
import os

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

# define agent class
class learning_agent(object):
    index2 = 0
    def __init__(self, a_name,env, policy_func, par):
        # Setup losses and stuff
        # ----------------------------------------
        self.env = env
        self.timesteps_per_actorbatch = par.timesteps_per_actorbatch
        self.optim_epochs= par.optim_epochs
        self.optim_stepsize = par.optim_stepsize
        self.optim_batchsize = par.optim_batchsize# optimization hypers
        self.gamma = par.gamma
        self.lam = par.lam # advantage estimation
        self.max_timesteps= par.max_timesteps
        self.max_episodes= par.max_episodes
        self.max_iters= par.max_iters 
        self.max_seconds= par.max_seconds  # time constraint
        self.callback= par.callback, # you can do anything in the callback, since it takes locals(), globals()
        self.adam_epsilon= par.adam_epsilon
        self.schedule=par.schedule # annealing for stepsize parameters (epsilon and adam)
        
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.pi = policy_func(a_name, self.ob_space, self.ac_space) # Construct network for new policy
        self.oldpi = policy_func("old"+a_name, self.ob_space, self.ac_space) # Network for old policy
        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        self.ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

        self.lrmult = tf.placeholder(name='lrmult'+a_name, dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
        self.clip_param = par.clip_param * self.lrmult # Annealed cliping parameter epislon
        
        obname = str('ob'+str(learning_agent.index2))
        learning_agent.index2 +=1
        self.ob = U.get_placeholder_cached(name=obname)
        self.ac = self.pi.pdtype.sample_placeholder([None])

        self.kloldnew = self.oldpi.pd.kl(self.pi.pd)
        self.ent = self.pi.pd.entropy()
        self.meankl = U.mean(self.kloldnew)
        self.meanent = U.mean(self.ent)
        self.pol_entpen = (-par.entcoeff) * self.meanent

        self.ratio = tf.exp(self.pi.pd.logp(self.ac) - self.oldpi.pd.logp(self.ac)) # pnew / pold
        surr1 = self.ratio * self.atarg # surrogate from conservative policy iteration
        surr2 = U.clip(self.ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.atarg #
        self.pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        self.vf_loss = U.mean(tf.square(self.pi.vpred - self.ret))
        self.total_loss = self.pol_surr + self.pol_entpen + self.vf_loss
        self.losses = [self.pol_surr, self.pol_entpen, self.vf_loss, self.meankl, self.meanent]
        self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        self.var_list = self.pi.get_trainable_variables()
        self.lossandgrad = U.function([self.ob, self.ac, self.atarg, self.ret, self.lrmult], 
                                      self.losses + [U.flatgrad(self.total_loss, self.var_list)])
        self.adam = MpiAdam(self.var_list, epsilon=self.adam_epsilon)

        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(self.oldpi.get_variables(), self.pi.get_variables())])
        self.compute_losses = U.function([self.ob, self.ac, self.atarg, self.ret, self.lrmult], self.losses)
        
        print(U.get_session())
        U.initialize()
        
        self.adam.sync()

  
    def learn(self):
    # Prepare for rollouts
    # ----------------------------------------
        seg_gen = traj_segment_generator(self.pi, self.env, self.timesteps_per_actorbatch, stochastic=True)
        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
        assert sum([self.max_iters>0, self.max_timesteps>0, 
                    self.max_episodes>0, self.max_seconds>0])==1, "Only one time constraint permitted"
        while True:
            if (timesteps_so_far >= self.max_timesteps) and self.max_timesteps:
                break
            elif (episodes_so_far >= self.max_episodes) and self.max_episodes :
                break
            elif (iters_so_far >= self.max_iters) and self.max_iters:
                break
            elif self.max_seconds and (time.time() - tstart >= self.max_seconds):
                break

            if self.schedule == 'constant':
                cur_lrmult = 1.0
            elif self.schedule == 'linear':
                cur_lrmult =  max(1.0 - float(timesteps_so_far) / self.max_timesteps, 0)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************"%iters_so_far)

            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, self.gamma, self.lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            self.ob, self.ac, self.atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            
            
            vpredbefore = seg["vpred"] # predicted value function before udpate
            self.atarg = (self.atarg - self.atarg.mean()) / self.atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=self.ob, ac=self.ac, atarg=self.atarg, vtarg=tdlamret), shuffle=not self.pi.recurrent)
            self.optim_batchsize = self.optim_batchsize or self.ob.shape[0]

            
            
            if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(self.ob) # update running mean/std for policy

            self.assign_old_eq_new() # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, self.loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(self.optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(self.optim_batchsize):
                    *newlosses, g = self.lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    self.adam.update(g, self.optim_stepsize * cur_lrmult) 
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(self.optim_batchsize):
                newlosses = self.compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)            
            meanlosses,_,_ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, self.loss_names):
                logger.record_tabular("loss_"+name, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            if MPI.COMM_WORLD.Get_rank()==0:
                logger.dump_tabular()


    def action_ev(self, obs):
        ac, vpred = self.pi.act(False, obs)
        return ac
    
    
    def restore(self, folder):
        U.load_state(folder+'/data')
        
    def save_data(self, folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            aaa = 1
            
        saver = U.tf.train.Saver()
        saver.save(U.get_session(), folder + '/data')
        
        