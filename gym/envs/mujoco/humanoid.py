import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass)) #[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)[0]
        z_before = mass_center(self.model, self.sim)[2] #self.sim.data.qpos[2]
        y_before = mass_center(self.model, self.sim)[1] #self.sim.data.qpos[1]
        all_before = self.sim.data.qpos
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[0]
        z_after = mass_center(self.model, self.sim)[2] #self.sim.data.qpos[2]
        y_after = mass_center(self.model, self.sim)[1] #self.sim.data.qpos[1]
        all_after = self.sim.data.qpos
        alive_bonus = 5.0
        data = self.sim.data
        lin_x_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        lin_z_cost = 0.25 * (z_after - z_before) / self.model.opt.timestep / (0.001 + min(1, pow((z_before - 0.3) / (0.7 - 0.3), 2))) #- 1 / self.model.opt.timestep * \
            #(1/(10 * np.abs(z_after - 0.5) + 0.1) + 1/(10 * np.abs(z_before - 0.5) + 0.1))
        lin_y_cost = 0.25 * (np.abs(y_after) - np.abs(y_before)) / self.model.opt.timestep
        #print(z_after)
        all_velocity_cost = np.mean(np.abs(all_after) - np.abs(all_before)) / self.model.opt.timestep + \
            (np.max(np.abs(all_after)) - np.max(np.abs(all_before))) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * min(1, pow((z_before-0.35)/(0.8-0.35), 4)) * np.square(data.ctrl).sum()  #z_before changes from ~0.93 in beg to ~0.35 when we terminate
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).mean()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_x_cost + lin_z_cost - lin_y_cost #- 0.1 * all_velocity_cost  - 0.01 * quad_ctrl_cost #- quad_impact_cost + alive_bonus
        #print("all_velocity_cost = ", all_velocity_cost)
        #print("quad_ctrl_cost = ", quad_ctrl_cost)
        #print("lin_x_cost = ", lin_x_cost)
        #print("lin_y_cost = ", lin_y_cost)
        #print("lin_z_cost = ", lin_z_cost)
        #print(mass_center(self.model, self.sim))
        qpos = self.sim.data.qpos
        done = bool((z_after < 0.35) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_x_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
