import sympy as sp
import numpy as np
from numpy import concatenate as concat

import gym
from gym import spaces
from env.Helicopter import Helicopter
from env.controller import Controller
from utils_main import save_files
import random


class HelicopterEnv(gym.Env):
    def __init__(self):
        print("changed reward tuned")
        self.Controller = Controller()
        self.U_input = [U1, U2, U3, U4] = sp.symbols("U1:5", real=True)
        self.x_state = [
            u_velocity,
            v_velocity,
            w_velocity,
            p_angle,
            q_angle,
            r_angle,
            fi_angle,
            theta_angle,
            si_angle,
            xI,
            yI,
            zI,
            a_flapping,
            b_flapping,
            c_flapping,
            d_flapping,
            uwind,
            vwind,
            wwind,
        ] = sp.symbols("x1:20", real=True)
        self.My_helicopter = Helicopter()
        self.My_controller = Controller()
        self.t = sp.symbols("t")
        self.symbolic_states_math, jacobian = self.My_helicopter.lambd_eq_maker(self.t, self.x_state, self.U_input)
        self.default_range = default_range = (-100, 100)
        self.velocity_range = velocity_range = (-100, 100)
        self.ang_velocity_range = ang_velocity_range = (-100, 100)
        self.ang_p_velocity_range = ang_p_velocity_range = (-100, 100)
        self.Ti, self.Ts, self.Tf = 0, 0.03, 8
        self.angle_range = angle_range = (-np.pi, np.pi)
        self.psi_range = psi_range = (-2 * np.pi, 2 * np.pi)
        self.observation_space_domain = {
            "u_velocity": velocity_range,
            "v_velocity": velocity_range,
            "w_velocity": velocity_range,
            "p_angle": ang_p_velocity_range,
            "q_angle": ang_velocity_range,
            "r_angle": ang_velocity_range,
            "fi_angle": angle_range,
            "theta_angle": angle_range,
            "si_angle": psi_range,
            "xI": default_range,
            "yI": default_range,
            "zI": default_range,
            "a_flapping": velocity_range,
            "b_flapping": velocity_range,
            "c_flapping": velocity_range,
            "d_flapping": velocity_range,
            # "t": (self.Ti, self.Tf),
            "delta_col": (-1000, 1000),
            "delta_lat": (-1000, 1000),
            "delta_lon": (-1000, 1000),
            "delta_ped": (-1000, 1000),
        }
        self.states_str = list(self.observation_space_domain.keys())
        self.low_obs_space = np.array(tuple(zip(*self.observation_space_domain.values()))[0], dtype=np.float32)
        self.high_obs_space = np.array(tuple(zip(*self.observation_space_domain.values()))[1], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_obs_space, high=self.high_obs_space, dtype=np.float32)
        self.default_act_range = default_act_range = (-0.3, 0.3)
        def_action = (-1, 1)
        lat_action = (-1, 1)
        self.action_space_domain = {
            "col_z": def_action,
            "col_w": def_action,
            "lon_x": def_action,
            "lon_u": def_action,
            "lon_q": def_action,
            "lon_eul_1": def_action,
            "lat_y": lat_action,
            "lat_v": lat_action,
            "lat_p": lat_action,
            "lat_eul_0": lat_action,
            "ped_r": def_action,
            "ped_eul_3": def_action,
        }
        self.low_action = np.array(tuple(zip(*self.action_space_domain.values()))[0], dtype=np.float32)
        self.high_action = np.array(tuple(zip(*self.action_space_domain.values()))[1], dtype=np.float32)
        self.low_action_space = self.low_action
        self.high_action_space = self.high_action
        self.action_space = spaces.Box(low=self.low_action_space, high=self.high_action_space, dtype=np.float32)
        self.min_reward = -13

        self.no_timesteps = int((self.Tf - self.Ti) / self.Ts)
        self.all_t = np.linspace(self.Ti, self.Tf, self.no_timesteps)
        self.counter = 0
        self.best_reward = float("-inf")
        self.longest_num_step = 0
        self.reward_check_time = 0.7 * self.Tf
        self.high_action_diff = 0.2
        obs_header = str(list(self.observation_space_domain.keys()))[1:-1]
        act_header = str(list(self.action_space_domain.keys()))[1:-1]
        self.header = (
            "time, "
            + act_header
            + ", "
            + obs_header[0:130]
            + ","
            + "uwind,"
            + "vwind,"
            + "wwind,"
            + obs_header[189:240]
            + "reward,"
            + "control_reward"
        )
        self.saver = save_files()
        self.reward_array = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.float32)
        self.reward_limit = [
            1.00e02,
            3.40e03,
            1.34e02,
            1.51e03,
            3.28e01,
            7.78e00,
            3.15e04,
            3.09e01,
            3.00e02,
            8.46e00,
            1.52e04,
            9.27e01,
        ]
        self.constant_dict = {
            "u": 0.0,
            "v": 0.0,
            "w": 0.0,
            "p": 1.0,
            "q": 1.0,
            "r": 0.0,
            "fi": 1.0,
            "theta": 1.0,
            "si": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "d": 0.0,
        }
        self.save_counter = 0
        self.longest_num_step = 0
        self.best_reward = float("-inf")
        self.diverge_counter = 0
        self.numTimeStep = int(self.Tf / self.Ts + 1)
        self.ifsave = 0
        self.low_control_input = [0.01, -0.1, -0.1, 0.01]
        self.high_control_input = [0.5, 0.1, 0.1, 0.5]
        self.cont_inp_dom = {"col": (-2.1, 2, 1), "lat": (-3.2, 3.2), "lon": (-3.5, 3.5), "ped": (-1.1, 1.1)}
        self.cont_str = list(self.cont_inp_dom.keys())
        self.initial_states = (
            np.array(
                (
                    3.70e-04,  # 0u
                    1.15e-02,  # 1v
                    4.36e-04,  # 2w
                    -5.08e-03,  # 3p
                    2.04e-04,  # 4q
                    2.66e-05,  # 5r
                    -1.08e-01,  # 6fi
                    1.01e-04,  # 7theta
                    -1.03e-03,  # 8si
                    -4.01e-05,  # 9x
                    -5.26e-02,  # 10y
                    -2.94e-04,  # 11z
                    -4.36e-06,  # 12a
                    -9.77e-07,  # 13b
                    -5.66e-05,  # 14c
                    7.81e-04,
                ),
                dtype=np.float32,
            )
            + 0.01
        )  # 15d
        # ran_ind = np.random.choice(3, size=1, replace=False)
        self.initial_states[9] = self.initial_states[9] + 1
        self.initial_states[10] = self.initial_states[10] + 1
        self.initial_states[11] = self.initial_states[11] + 1
        print("initial_stateeeeeeeeeeeeeees")
        self.wind1 = np.array((0, 0, 0))  # (-1) ** np.random.choice([0, 1], 3) * 0 + 0.25 * (np.random.random(3) - 0.5)
        self.jk = 1

    def reset(self):
        # initialization
        self.t = 0
        self.all_obs = np.zeros((self.no_timesteps, len(self.high_obs_space)))
        self.all_actions = np.zeros((self.no_timesteps, len(self.high_action_space)))
        self.all_control = np.zeros((self.no_timesteps, 4))
        self.all_rewards = np.zeros((self.no_timesteps, 1))
        self.control_rewards = np.zeros((self.no_timesteps, 1))
        self.control_input = np.array((0, 0, 0, 0), dtype=np.float32)
        self.jj = 0
        self.counter = 0
        # Yd, Ydotd, Ydotdotd, Y, Ydot = self.My_controller.Yposition(0, self.current_states)

        # self.current_states = self.initial_states  # * (1 + s * a)
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.wind = (
            self.wind1  # + ((-1) ** np.random.choice([0, 1], 3) * 1 + 0.025 * (np.random.random(3) - 0.5)) / self.jk
        )
        self.jk = self.jk + 0.001
        self.current_states = concat((self.initial_states, self.wind), axis=0)  # * (1 + s * a)

        self.observation = self.observation_function()
        self.done = False
        self.integral_error = 0
        return self.observation

    def action_wrapper(self, current_action, obs) -> np.array:
        self.normilized_actions = current_action
        un_act = (current_action + 1) * (self.high_action - self.low_action) / 2 + self.low_action
        self.all_actions[self.counter] = self.normilized_actions  # unnormalized_action
        self.control_input[0] = un_act[0] * 5 * obs[11] + un_act[1] * 5 * obs[2]
        self.control_input[2] = (
            un_act[2] * 5 * obs[9] + un_act[3] * 5 * obs[0] + un_act[4] * 5 * obs[4] + un_act[5] * obs[7]
        )
        self.control_input[1] = (
            un_act[6] * 5 * obs[10] + un_act[7] * 5 * obs[1] + un_act[8] * 5 * obs[3] + un_act[9] * obs[6]
        )
        self.control_input[3] = un_act[10] * 5 * obs[5] + un_act[11] * 5 * obs[8]
        self.control_input[0] = 2.1167 * self.control_input[0] + 0.1
        self.control_input[1] = 2.03125 * self.control_input[1]
        self.control_input[2] = 2.02857 * self.control_input[2]
        self.control_input[3] = 2.2227 * self.control_input[3] + 0.18

        self.all_control[self.counter] = self.control_input

    def find_next_state(self) -> list:
        current_t = self.Ts * self.counter
        # self.control_input = self.Controller.Controller_model(
        #     self.current_states[0:16],
        #     current_t,
        #     # action=[1.09e-01, -1.78e-02, -2.20e-05, 0.19, 0.4, 4.1, 4.1, 1.5, 1.5, 0.5, 1, 0.7, 0.5, 0.4, 1, 0.6, 0.5],
        # )
        self.current_states[0:19] = self.My_helicopter.RK45(
            current_t,
            self.current_states[0:19],
            self.symbolic_states_math,
            self.Ts,
            self.control_input,
        )
        self.current_states[16:19] = self.wind  # = self.wind + 0.005 * (np.random.random(3) - 0.5)

    def observation_function(self) -> list:
        self.observation = concat((self.current_states[0:16], self.control_input), axis=0)
        self.all_obs[self.counter] = concat((self.current_states[0:16], self.control_input), axis=0)
        for iii in range(20):
            current_range = self.observation_space_domain[self.states_str[iii]]
            self.observation[iii] = (
                2 * (self.observation[iii] - current_range[0]) / (current_range[1] - current_range[0]) - 1
            )
        return self.observation

    def reward_function(self, observation, rew_cof=[650, 0.4, 1.5]) -> float:
        error = -rew_cof[0] * (np.linalg.norm(observation[9:12].reshape(3), 1))
        # print("error", error)
        if all(abs(self.current_states[9:12])) < 0.1:
            error = error + 1 - abs(observation[8])
        reward = error.copy()
        self.control_rewards[self.counter] = error
        self.integral_error = 0.025 * self.control_rewards[self.counter] + self.integral_error
        # print("integral", self.integral_error)
        reward += self.integral_error
        x = self.current_states[9]
        y = self.current_states[10]
        si = self.current_states[8]
        si_d_angle = 0
        if x >= 0:
            if y > 0:
                si_d_angle = np.arcsin(-x / np.sqrt(x ** 2 + y ** 2)) - np.pi / 2

            else:
                si_d_angle = np.arcsin(x / np.sqrt(x ** 2 + y ** 2)) + np.pi / 2
        else:
            if y >= 0:
                si_d_angle = -np.arcsin(x / np.sqrt(x ** 2 + y ** 2)) - np.pi / 2
            else:
                si_d_angle = np.arcsin(x / np.sqrt(x ** 2 + y ** 2)) + np.pi / 2
        if si_d_angle > np.pi:
            si_d_angle -= 2 * np.pi
        if si_d_angle <= -np.pi:
            si_d_angle += 2 * np.pi
        if -np.pi <= si <= np.pi:
            si_ini_er = si - si_d_angle
            si_error = min(abs(si_ini_er), abs(si_ini_er + 2 * np.pi), abs(si_ini_er - 2 * np.pi)) / self.psi_range[1]
        elif si > np.pi:
            si_ini_er = si - si_d_angle - 2 * np.pi
            si_error = min(abs(si_ini_er), abs(si_ini_er + 2 * np.pi), abs(si_ini_er - 2 * np.pi)) / self.psi_range[1]
        else:
            si_ini_er = si - si_d_angle + 2 * np.pi
            si_error = min(abs(si_ini_er), abs(si_ini_er + 2 * np.pi), abs(si_ini_er - 2 * np.pi)) / self.psi_range[1]
        reward -= si_error
        reward += 30000 / self.numTimeStep
        # print(4500 / self.numTimeStep)
        reward -= rew_cof[1] * sum(abs(self.control_input - self.all_control[self.counter - 1, :]))
        # print("cntrl_d", -rew_cof[1] * sum(abs(self.control_input - self.all_control[self.counter - 1, :])))
        reward -= rew_cof[2] * np.linalg.norm(self.control_input, 2)
        # print("contrl", -rew_cof[2] * np.linalg.norm(self.control_input, 2))
        self.all_rewards[self.counter] = reward
        # print("rew", reward)
        return reward

    def check_diverge(self, reward) -> bool:
        bool_1 = any(np.isnan(self.current_states))
        bool_2 = any(np.isinf(self.current_states))
        if bool_1 or bool_2:
            self.jj = 1
            print("state_inf_nan_diverge")
            self.observation = self.all_obs[self.counter - 1]
            reward = self.min_reward - 100
            return True, reward
        try:
            if any(np.isnan(reward)) or any(np.isnan(reward)):
                reward = self.min_reward - 100
                return True, reward
        except OverflowError:
            pass
        for i in range(12):
            if (abs(self.all_obs[self.counter, i])) > self.high_obs_space[i]:
                self.saver.diverge_save(self.observation_space_domain, i)
                self.jj = 1

        if self.jj == 1:
            return True, reward
        if self.counter >= self.no_timesteps - 1:  # number of timesteps
            return True, reward
        # after self.reward_check_time it checks whether or not the reward is decreasing

        return False, reward

    def done_jobs(self) -> None:
        counter = self.counter
        self.save_counter += 1
        current_total_reward = sum(self.all_rewards)
        if self.save_counter >= 1000:
            print("current_total_reward: ", current_total_reward)
            self.save_counter = 0
            self.saver.reward_step_save(self.best_reward, self.longest_num_step, current_total_reward, counter)
        if counter >= self.longest_num_step:
            self.longest_num_step = counter
        if current_total_reward >= self.best_reward * 1.01 and sum(self.all_rewards) != 0:
            self.best_reward = current_total_reward
            ii = self.counter + 1
            self.saver.best_reward_save(
                self.all_t[0:ii],
                self.all_actions[0:ii],
                self.all_obs[0:ii],
                self.all_rewards[0:ii],
                self.control_rewards[0:ii],
                self.header,
            )

    def step(self, current_action):
        self.action_wrapper(current_action, self.observation)
        try:
            self.find_next_state()
        except OverflowError or ValueError or IndexError:
            self.jj = 1
        self.observation = self.observation_function()
        reward = self.reward_function(self.observation)
        self.done, reward = self.check_diverge(reward)
        if self.jj == 1:
            reward -= self.min_reward
        if self.done:
            self.done_jobs()
        self.counter += 1
        # self.make_constant(list(self.constant_dict.values()))
        return self.observation, reward, self.done, {}

    def make_constant(self, true_list):
        for i in range(len(true_list)):
            if i == 1:
                self.current_states[i] = self.initial_states[i]

    def close(self):
        return None