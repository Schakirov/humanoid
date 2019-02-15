import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'swimmer_my.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001 * 0.0001
        xposbefore = self.sim.data.qpos[0]
        yposbefore = self.sim.data.qpos[1]
        xposbefore_sw2 = self.sim.data.qpos[9+0]
        yposbefore_sw2 = self.sim.data.qpos[9+1]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        yposafter = self.sim.data.qpos[1]
        xposafter_sw2 = self.sim.data.qpos[9+0]
        yposafter_sw2 = self.sim.data.qpos[9+1]
        x_dist_increase_velocity = (abs(xposbefore_sw2 - xposbefore) - abs(xposafter_sw2 - xposafter)) / self.dt
        y_dist_increase_velocity = (abs(yposbefore_sw2 - yposbefore) - abs(yposafter_sw2 - yposafter)) / self.dt
        reward_fwd = 0 #(xposafter - xposbefore) / self.dt #- (abs(yposafter) - abs(yposbefore)) * 10 * abs(yposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl + x_dist_increase_velocity + y_dist_increase_velocity
        #print(reward_fwd, '\n', reward_ctrl, '\n', x_dist_increase_velocity, '\n', y_dist_increase_velocity, '\n\n')
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        #print(qpos, '\n\n\n', qvel)
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
