"""Real Torch optimizer/model round trips; no simulator or private checkpoints."""
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from regrind.utils.arm_transfer import prepare_transfer, install_transfer_time_limit, TransferTimeLimitReached


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(67,12)
        self.register_buffer('normalizer_mean',torch.ones(67))
        self.distribution=torch.nn.Parameter(torch.full((12,),.5))

    def forward(self, obs):
        return self.linear(obs['policy']-self.normalizer_mean)


def runner():
    actor=Actor();critic=torch.nn.Linear(94,1)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1e-4)
    alg=SimpleNamespace(_raw_actor=actor,_raw_critic=critic,optimizer=optimizer,learning_rate=1e-4,
                        eval_mode=lambda: (actor.eval(),critic.eval()))
    env=SimpleNamespace(num_envs=1,unwrapped=SimpleNamespace(common_step_counter=0),
        get_observations=lambda:dict(policy=torch.ones(1,67),critic=torch.ones(1,94)))
    result=SimpleNamespace(alg=alg,env=env,device='cpu',current_learning_iteration=0,logger=SimpleNamespace())
    def load(path,load_cfg):
        saved=torch.load(path,weights_only=False)
        actor.load_state_dict(saved['actor_state_dict']);critic.load_state_dict(saved['critic_state_dict'])
        if load_cfg['optimizer']:optimizer.load_state_dict(saved['optimizer_state_dict'])
        if load_cfg['iteration']:result.current_learning_iteration=saved['iter']
    result.load=load
    return result


class TransferCheckpointTest(unittest.TestCase):
    def test_time_budget_stops_only_after_logging_completed_update(self):
        calls=[]
        r=SimpleNamespace(logger=SimpleNamespace(log=lambda **kw:calls.append(kw)))
        with patch('regrind.utils.arm_transfer.time.perf_counter',side_effect=[10.,14.,15.1]):
            install_transfer_time_limit(r,5.)
            r.logger.log(it=0)
            with self.assertRaises(TransferTimeLimitReached):r.logger.log(it=1)
        self.assertEqual(calls,[{'it':0},{'it':1}])
        for invalid in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):install_transfer_time_limit(r,invalid)

    def test_fresh_transfer_loads_models_not_old_optimizer_or_iteration(self):
        origin=runner()
        origin.alg._raw_actor.normalizer_mean.fill_(2)
        origin.alg.optimizer.param_groups[0]['lr']=.003
        origin.alg._raw_actor({'policy':torch.ones(1,67)}).sum().backward()
        origin.alg.optimizer.step()
        saved=dict(actor_state_dict=origin.alg._raw_actor.state_dict(),
                   critic_state_dict=origin.alg._raw_critic.state_dict(),
                   optimizer_state_dict=origin.alg.optimizer.state_dict(),iter=9999,infos=None)
        with tempfile.TemporaryDirectory() as directory,patch('regrind.utils.arm_transfer.controller_contract',return_value={'fixed':True}):
            path=Path(directory)/'floating.pt';torch.save(saved,path)
            current=runner();info=prepare_transfer(current,path)
            self.assertEqual(info['deterministic_action_max_difference'],0)
            self.assertEqual(current.current_learning_iteration,0)
            self.assertFalse(current.alg.optimizer.state)
            self.assertEqual(current.alg.optimizer.param_groups[0]['lr'],1e-4)
            self.assertTrue(torch.equal(current.alg._raw_actor.normalizer_mean,torch.full((67,),2.)))
            with self.assertRaisesRegex(ValueError,'Resume requires'):
                prepare_transfer(runner(),path,resume=True)

    def test_transfer_resume_restores_optimizer_lr_and_next_iteration(self):
        import numpy as np,random
        origin=runner()
        origin.alg._raw_actor({'policy':torch.zeros(1,67)}).sum().backward();origin.alg.optimizer.step()
        origin.alg.optimizer.param_groups[0]['lr']=2e-5
        saved=dict(actor_state_dict=origin.alg._raw_actor.state_dict(),critic_state_dict=origin.alg._raw_critic.state_dict(),
            optimizer_state_dict=origin.alg.optimizer.state_dict(),iter=99,
            infos={'arm_transfer':dict(controller_contract={'fixed':True},common_step_counter=2400,
                total_transitions=38400,training_wall_seconds=5,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy_rng=np.random.get_state(),python_rng=random.getstate())})
        with tempfile.TemporaryDirectory() as directory,patch('regrind.utils.arm_transfer.controller_contract',return_value={'fixed':True}):
            path=Path(directory)/'transfer.pt';torch.save(saved,path)
            current=runner();prepare_transfer(current,path,resume=True)
            self.assertTrue(current.alg.optimizer.state)
            self.assertEqual(current.alg.learning_rate,2e-5)
            self.assertEqual(current.current_learning_iteration,100)
            self.assertEqual(current.env.unwrapped.common_step_counter,2400)
            with self.assertRaisesRegex(ValueError,'Use --resume'):
                prepare_transfer(runner(),path)


class TransferComparisonTest(unittest.TestCase):
    def test_rejects_changed_controller_or_measured_initial_state(self):
        from tools.arm_diagnostics.analyze_transfer_recovery import compare_transfer
        state = {key:[0.] for key in ('all_q','all_v','robot_root','object_state','wrist_pos',
            'wrist_quat','wrist_velocity','hand_q','hand_v','follower_q','follower_v','phase')}
        meta = dict(mode='legacy',stage='grasp',ends=[{}]*20,frozen_policy_verified=True,
            transfer_controller_contract={'physics_dt':1/120},initial_states=[state]*20)
        for key in ('reference','physics_dt','control_dt','limits','gains','gravity','robot_spawn','joint_names','state_bank'):
            meta[key] = 'same'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('before','after'):
                (root/name).mkdir();(root/name/'metadata.json').write_text(json.dumps(meta))
            changed = json.loads(json.dumps(meta))
            changed['transfer_controller_contract']['physics_dt'] = 1/60
            (root/'after/metadata.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Changed paired evaluation condition'):
                compare_transfer(root/'before',root/'after',root/'out')
            changed = json.loads(json.dumps(meta))
            changed['initial_states'][4]['all_v'] = [.1]
            (root/'after/metadata.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(AssertionError,'initial state 4: all_v'):
                compare_transfer(root/'before',root/'after',root/'out')
            self.assertFalse((root/'out').exists())
