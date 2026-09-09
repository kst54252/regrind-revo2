"""Opt-in transfer initialization/audit around the installed RSL-RL PPO runner.

No policy, PPO, IK or actuator implementation lives here. Simulation state is
not claimed to be checkpoint-serializable: resume resets a new physical episode.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array(value):
    value = getattr(value, 'torch', value)
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value


def controller_contract(env):
    """Numerical/controller semantics shared by train and the existing evaluator."""
    root = env.action_manager.get_term('root_pose')
    hand = env.action_manager.get_term('joint_pos')
    robot = env.scene['robot']
    keys = ('base_action_source', 'scale_pos', 'scale_rot', 'raw_clip',
            'interpolation_substeps', 'velocity_target_mode', 'transfer_reset_sync',
            'model_config_path', 'base_position', 'base_quaternion_xyzw',
            'position_tolerance_m', 'orientation_tolerance_rad', 'position_weight', 'max_nfev')
    if type(root).__name__ == 'SimpleMountedWrist':
        keys = ('base_action_source','scale_pos','scale_rot','raw_clip','response_tau',
                'velocity_path','response_at_physics','fast_ik','ik_policy_rate',
                'velocity_bounded_ik','ik_acceleration_limit','body_name','command_name')
    result = dict(arm_class=type(root).__name__, arm={k:getattr(root.cfg,k) for k in keys},
                  physics_dt=env.physics_dt, policy_dt=env.step_dt,
                  gravity=env.cfg.sim.gravity, joint_names=list(robot.joint_names),
                  hand_scale=hand.cfg.scale, hand_clip=hand.cfg.raw_clip,
                  hand_base=hand.cfg.base_action_source,
                  observation_terms=env.observation_manager.active_terms,
                  reference_sha256=sha256(env.command_manager.get_term('reference').reference.path))
    for name in ('joint_stiffness', 'joint_damping', 'joint_effort_limits', 'joint_vel_limits', 'soft_joint_pos_limits'):
        result[name] = array(getattr(robot.data, name))[0]
    contact = getattr(robot.cfg.spawn, 'revo2_contact_spec', None)
    if contact is not None:
        result['fingertip_contact'] = contact
    if type(root).__name__ == 'SimpleMountedWrist':
        from regrind.assets import REGRIND_PROJECT_ROOT
        from regrind.workcell import ROBOT_MOUNT_POSITION, ROBOT_MOUNT_QUATERNION_XYZW
        result['model_sha256']=sha256(REGRIND_PROJECT_ROOT/'tools/rb3_revo2_ik/rb3_model.json')
        result['base_position']=ROBOT_MOUNT_POSITION
        result['base_quaternion_xyzw']=ROBOT_MOUNT_QUATERNION_XYZW
    return json.loads(json.dumps(result))


def prepare_transfer(runner, source, *, resume=False):
    source = str(Path(source).resolve())
    checkpoint = torch.load(source, map_location=runner.device, weights_only=False)
    prior = (checkpoint.get('infos') or {}).get('arm_transfer')
    if resume and not prior:
        raise ValueError('Resume requires a transfer checkpoint with arm_transfer metadata; use --transfer-init for floating')
    if not resume and prior:
        raise ValueError('Use --resume for an existing transfer run, not --transfer-init')
    contract = controller_contract(runner.env.unwrapped)
    source_cfg = Path(source).parent/'params/env.yaml'
    source_terms = None
    if not resume and source_cfg.is_file():
        import yaml
        # BaseLoader reads Python-tagged config values as data; no object construction.
        config = yaml.load(source_cfg.read_text(), Loader=yaml.BaseLoader)
        source_terms = {group:[name for name,term in config['observations'][group].items()
                             if isinstance(term,dict) and 'func' in term]
                        for group in ('policy','critic')}
        for group,names in source_terms.items():
            if names != list(contract['observation_terms'][group]):
                raise ValueError(f'Checkpoint observation order differs: {group}')
        for name in ('scale_pos','scale_rot'):
            if float(config['actions']['root_pose'][name]) != float(contract['arm'][name]):
                raise ValueError(f'Checkpoint wrist residual scale differs: {name}')
    if prior and prior['controller_contract'] != contract:
        raise ValueError('Transfer resume controller/reference contract differs from checkpoint')
    obs = runner.env.get_observations()
    if tuple(obs['policy'].shape) != (runner.env.num_envs,67) or tuple(obs['critic'].shape) != (runner.env.num_envs,94):
        raise ValueError('Expected checkpoint-compatible actor 67 / critic 94')
    oracle = deepcopy(runner.alg._raw_actor).eval()
    oracle.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    runner.load(source, load_cfg=dict(actor=True,critic=True,optimizer=resume,iteration=resume,rnd=False))
    runner.alg.eval_mode()
    with torch.inference_mode():
        expected = oracle(obs)
        actual = runner.alg._raw_actor(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if actual.shape != (runner.env.num_envs,12) or not torch.isfinite(actual).all():
        raise ValueError('Invalid loaded deterministic action')
    for prefix,model in (('actor',runner.alg._raw_actor),('critic',runner.alg._raw_critic)):
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,checkpoint[prefix+'_state_dict'][key],rtol=0,atol=0)
    if resume:
        # Installed runner stores the last completed update, not the next one.
        runner.current_learning_iteration = int(checkpoint['iter']) + 1
        runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]['lr'])
        runner.env.unwrapped.common_step_counter = prior['common_step_counter']
        runner.logger.tot_timesteps = prior['total_transitions']
        runner.logger.tot_time = prior['training_wall_seconds']
        torch.set_rng_state(prior['torch_rng'].cpu())
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in prior['cuda_rng']])
        np.random.set_state(prior['numpy_rng'])
        random.setstate(prior['python_rng'])
    elif runner.alg.optimizer.state or runner.current_learning_iteration != 0:
        raise ValueError('Initial transfer must have a fresh optimizer and zero iteration')
    info = dict(mode='resume' if resume else 'initialize', source=source,
                source_sha256=sha256(source), source_iteration=int(checkpoint['iter']),
                initial_iteration=runner.current_learning_iteration,
                learning_rate=runner.alg.learning_rate,
                optimizer_lr=[g['lr'] for g in runner.alg.optimizer.param_groups],
                deterministic_action_max_difference=float((actual-expected).abs().max()),
                actor_shape=list(obs['policy'].shape),critic_shape=list(obs['critic'].shape),
                action_shape=list(actual.shape),controller_contract=contract,
                source_observation_terms=source_terms,
                normalizer='Loaded exactly; installed PPO updates normalization during training; evaluation frozen',
                resume_physics='Fresh reference reset; hidden PhysX contacts and running episodes are not restored')
    print('[arm transfer initialization]',json.dumps(info),flush=True)
    return info


class TransferAudit:
    """Fail-fast finite checks, measured PPO/IK throughput, and resume metadata."""
    def __init__(self, runner, wrapper, source, log_dir, info):
        self.runner, self.env, self.info = runner, wrapper.unwrapped, info
        self.source = source
        self.out = Path(log_dir)
        self.out.mkdir(parents=True,exist_ok=True)
        (self.out/'transfer_initialization.json').write_text(json.dumps(info,indent=2)+'\n')
        self.stream = (self.out/'transfer_updates.jsonl').open('x')
        self.start = self.last = time.perf_counter()
        self.steps = self.resets = self.ik_calls = 0
        self.ik_seconds = 0.
        self.optimizer_steps = 0
        self.gradient_max = 0.
        self.initial_weights = {k:v.detach().clone() for k,v in runner.alg._raw_actor.named_parameters()}
        self.rewards = []
        self.root = self.env.action_manager.get_term('root_pose')
        # Time the actual solve entry, including warm-first/fallback processing.
        owner = self.root if hasattr(self.root,'solve') else self.root._kinematics
        method = 'solve' if hasattr(self.root,'solve') else 'inverse'
        inverse = getattr(owner,method)
        def timed_inverse(*args,**kwargs):
            begin=time.perf_counter()
            result=inverse(*args,**kwargs)
            self.ik_seconds+=time.perf_counter()-begin
            self.ik_calls+=1
            return result
        setattr(owner,method,timed_inverse)
        step=wrapper.step
        def measured_step(actions):
            if not torch.isfinite(actions).all(): raise ValueError('Nonfinite policy action')
            result=step(actions)
            obs,reward,done,_=result
            for value in (*obs.values(),reward):
                if not torch.isfinite(value).all(): raise ValueError('Nonfinite observation/reward')
            self.steps+=1
            self.resets+=int(done.sum())
            self.rewards.append(float(reward.mean()))
            return result
        wrapper.step=measured_step
        optimizer_step=runner.alg.optimizer.step
        def checked_optimizer_step(*args,**kwargs):
            for group in runner.alg.optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all(): raise ValueError('Nonfinite PPO gradient')
                        self.gradient_max=max(self.gradient_max,float(param.grad.abs().max()))
            result=optimizer_step(*args,**kwargs)
            self.optimizer_steps+=1
            return result
        runner.alg.optimizer.step=checked_optimizer_step
        update=runner.alg.update
        def checked_update():
            losses=update()
            if not all(np.isfinite(float(v)) for v in losses.values()): raise ValueError('Nonfinite PPO loss')
            for model in (runner.alg._raw_actor,runner.alg._raw_critic):
                if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                    raise ValueError('Nonfinite policy/normalizer')
            delta=max(float((v-self.initial_weights[k]).abs().max()) for k,v in runner.alg._raw_actor.named_parameters())
            now=time.perf_counter()
            row=dict(update_count=len(self.rows)+1,transitions=self.steps*wrapper.num_envs,
                     num_envs=wrapper.num_envs,elapsed_seconds=now-self.start,
                     update_seconds=now-self.last,losses={k:float(v) for k,v in losses.items()},
                     reward_mean=float(np.mean(self.rewards)),reset_count=self.resets,
                     optimizer_steps=self.optimizer_steps,gradient_max=self.gradient_max,
                     actor_weight_max_change=delta,ik_calls=self.ik_calls,ik_seconds=self.ik_seconds,
                     learning_rate=runner.alg.learning_rate,finite=True)
            self.rows.append(row);self.stream.write(json.dumps(row)+'\n');self.stream.flush()
            self.last=now;self.rewards.clear()
            print('[arm transfer update]',json.dumps(row),flush=True)
            return losses
        self.rows=[]
        runner.alg.update=checked_update
        save=runner.save
        def save_transfer(path,infos=None):
            data=dict(infos or {})
            data['arm_transfer']=dict(controller_contract=info['controller_contract'],
                common_step_counter=int(self.env.common_step_counter),
                total_transitions=runner.logger.tot_timesteps,training_wall_seconds=runner.logger.tot_time,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),initialization=info)
            save(path,infos=data)
        runner.save=save_transfer

    def finish(self):
        self.stream.close()
        if sha256(self.source)!=self.info['source_sha256']: raise ValueError('Source checkpoint changed')
        if not self.rows or self.rows[-1]['actor_weight_max_change']==0 or self.gradient_max==0:
            raise ValueError('No verified policy learning occurred')
        result=dict(self.rows[-1],source_unchanged=True,initialization=self.info)
        (self.out/'transfer_summary.json').write_text(json.dumps(result,indent=2)+'\n')


class TransferTimeLimitReached(Exception):
    """Normal stop only after a completed PPO update has been logged."""


def install_transfer_time_limit(runner, seconds):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Training wall-time budget must be positive and finite')
    started=time.perf_counter();original=runner.logger.log
    def log_then_check(*args,**kwargs):
        result=original(*args,**kwargs)
        if time.perf_counter()-started >= seconds:
            raise TransferTimeLimitReached()
        return result
    runner.logger.log=log_then_check
