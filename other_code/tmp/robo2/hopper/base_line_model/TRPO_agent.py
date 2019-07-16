from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager
import os

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

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
            _, vpred = pi.act(stochastic, ob)            
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
    
class TRPO_agent_new(object):
    index2 = 0
    def __init__(self, a_name,env, policy_func,par):
        
        self.env = env
        self.timesteps_per_batch = par.timesteps_per_batch
        self.max_kl = par.max_kl
        self.cg_iters = par.cg_iters
        self.gamma = par.gamma
        self.lam = par.lam # advantage estimation
        self.entcoeff = par.entcoeff
        self.cg_damping=par.cg_damping
        self.vf_stepsize=par.vf_stepsize
        self.vf_iters = par.vf_iters
        self.max_timesteps= par.max_timesteps
        self.max_episodes= par.max_episodes
        self.max_iters= par.max_iters 
        self.callback= par.callback, # you can do anything in the callback, since it takes locals(), globals()

        self.nworkers = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        np.set_printoptions(precision=3)    
    # Setup losses and stuff
    # ----------------------------------------
        self.ob_space = self.env.observation_space
        self.ac_space = self.env.action_space
        self.pi = policy_func(a_name, self.ob_space, self.ac_space)
        self.oldpi = policy_func("oldpi"+a_name, self.ob_space, self.ac_space)
        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        self.ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

        self.ob = U.get_placeholder_cached(name="ob"+str(TRPO_agent_new.index2))
        self.ac = self.pi.pdtype.sample_placeholder([None])
        
        
        
        self.kloldnew = self.oldpi.pd.kl(self.pi.pd)
        self.ent = self.pi.pd.entropy()
        meankl = U.mean(self.kloldnew)
        meanent = U.mean(self.ent)
        entbonus = self.entcoeff * meanent

        self.vferr = U.mean(tf.square(self.pi.vpred - self.ret))

        ratio = tf.exp(self.pi.pd.logp(self.ac) - self.oldpi.pd.logp(self.ac)) # advantage * pnew / pold
        surrgain = U.mean(ratio * self.atarg)

        optimgain = surrgain + entbonus
        self.losses = [optimgain, meankl, entbonus, surrgain, meanent]
        self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

        self.dist = meankl

        all_var_list = self.pi.get_trainable_variables()
        
        var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
        self.vfadam = MpiAdam(vf_var_list)

        self.get_flat = U.GetFlat(var_list)
        self.set_from_flat = U.SetFromFlat(var_list)
        self.klgrads = tf.gradients(self.dist, var_list)
        self.flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan"+str(TRPO_agent_new.index2))
        
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        self.tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            self.tangents.append(tf.reshape(self.flat_tangent[start:start+sz], shape))
            start += sz
    
        self.gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(self.klgrads, self.tangents)]) #pylint: disable=E1111
        self.fvp = U.flatgrad(self.gvp, var_list)

        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(self.oldpi.get_variables(), self.pi.get_variables())])
        
        self.compute_losses = U.function([self.ob, self.ac, self.atarg], self.losses)
        self.compute_lossandgrad = U.function([self.ob, self.ac, self.atarg], self.losses + [U.flatgrad(optimgain, var_list)])
        self.compute_fvp = U.function([self.flat_tangent, self.ob, self.ac, self.atarg], self.fvp)
        self.compute_vflossandgrad = U.function([self.ob, self.ret], U.flatgrad(self.vferr, vf_var_list))
        
        
        TRPO_agent_new.index2 +=1
        U.initialize()
        self.th_init = self.get_flat()
        MPI.COMM_WORLD.Bcast(self.th_init, root=0)
        self.set_from_flat(self.th_init)
        self.vfadam.sync()
        print("Init param sum", self.th_init.sum(), flush=True)
        
    @contextmanager
    def timed(self,msg):
        if self.rank == 0:                               ##################################
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield
    
    def allmean(self,x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= self.nworkers                                   ####################################
        return out

        
    def fisher_vector_product(self,p):
        return self.allmean(self.compute_fvp(p, *self.fvpargs)) + self.cg_damping * p
    
    
    def learn(self):
    
    # Prepare for rollouts
    # ----------------------------------------
        seg_gen = traj_segment_generator(self.pi, self.env, self.timesteps_per_batch, stochastic=True)

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

        assert sum([self.max_iters>0, self.max_timesteps>0, self.max_episodes>0])==1

        while True:        
        
            if self.max_timesteps and timesteps_so_far >= self.max_timesteps:
                break
            elif self.max_episodes and episodes_so_far >= self.max_episodes:
                break
            elif self.max_iters and iters_so_far >= self.max_iters:
                break
            logger.log("********** Iteration %i ************"%iters_so_far)

            with self.timed("sampling"):
                seg = seg_gen.__next__()
            
            add_vtarg_and_adv(seg, self.gamma, self.lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            self.ob, self.ac, self.atarg, self.tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            self.vpredbefore = seg["vpred"] # predicted value function before udpate
            self.atarg = (self.atarg - self.atarg.mean()) / self.atarg.std() # standardized advantage function estimate

            if hasattr(self.pi, "ret_rms"): self.pi.ret_rms.update(self.tdlamret)
            if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(self.ob) # update running mean/std for policy

            args = seg["ob"], seg["ac"], self.atarg
            self.fvpargs = [arr[::5] for arr in args]
        
            
        
            self.assign_old_eq_new() # set old parameter values to new parameter values
            with self.timed("computegrad"):
                *lossbefore, g = self.compute_lossandgrad(*args)
            
            
            lossbefore = self.allmean(np.array(lossbefore))
            g = self.allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
            
                with self.timed("cg"):
                    stepdir = cg(self.fisher_vector_product, g, cg_iters=self.cg_iters, verbose=self.rank==0)
                
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(self.fisher_vector_product(stepdir))
                lm = np.sqrt(shs / self.max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = self.get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    self.set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = self.allmean(np.array(self.compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > self.max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    self.set_from_flat(thbefore)
                if self.nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum())) # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
                    
            for (lossname, lossval) in zip(self.loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

            with self.timed("vf"):

                for _ in range(self.vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]), 
                    include_final_partial_batch=False, batch_size=64):
                        g = self.allmean(self.compute_vflossandgrad(mbob, mbret))
                        self.vfadam.update(g, self.vf_stepsize)

            logger.record_tabular("ev_tdlam_before", explained_variance(self.vpredbefore, self.tdlamret))

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

            if self.rank==0:
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
            if e.errno != errno.EEXIST:
                raise
                
        saver = U.tf.train.Saver()
        saver.save(U.get_session(), folder + '/data')