from cProfile import run
import re
from turtle import st
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import random
import matplotlib.pyplot as plt
import time

import numpy as np
from itertools import combinations

import math

import config as cf

from car_road import Map

from vehicle2 import Car
from vehicle2 import is_invalid_road
#from vehicle_stanley import evaluate
import os
import time
from stable_baselines3.common.evaluation import evaluate_policy


import json
from stable_baselines3 import PPO, DQN, A2C

from stable_baselines3.common.env_checker import check_env


from stable_baselines3.common.callbacks import CheckpointCallback

import wandb

def get_points_from_states(states):
    map = Map(
    cf.model["map_size"])
    tc = states
    for state in tc:
        action = state[0]
        if action == 0:
            done = map.go_straight(state[1])
            if not(done):
                break
        elif action == 2:
            done = map.turn_left(state[2])
            if not(done):
                break
        elif action == 1:
            done = map.turn_right(state[2])
            if not(done):
                break
        else:
            print("ERROR, invalid action")

    points = map.road_points_list
    return points

def generate_random_road():

    invalid = True

    while invalid == True:

        map = Map(
            cf.model["map_size"])

        #actions = list(range(0, 3))
        lengths = list(range(cf.model["min_len"], cf.model["max_len"]))
        angles = list(range(cf.model["min_angle"], cf.model["max_angle"]))
        done = False

        while not done:
            length = np.random.choice(lengths)
            done = not(map.go_straight(length))

        scenario = map.scenario[:-1]
        road = get_points_from_states(scenario)
        invalid = is_invalid_road(road)
    return scenario



def generate_random_state():
    
    map = Map(
    cf.model["map_size"])

    actions = list(range(0, 3))
    lengths = list(range(cf.model["min_len"], cf.model["max_len"]))
    angles = list(range(cf.model["min_angle"], cf.model["max_angle"]))
    done = False

    #while not done:
    action = np.random.choice(actions)
    if action == 0:
        length = np.random.choice(lengths)
        done = not(map.go_straight(length))
    elif action == 1:
        angle = np.random.choice(angles)
        done = not(map.turn_right(angle))
    elif action == 2:
        angle = np.random.choice(angles)
        done = not(map.turn_left(angle))

    #road_points = map.road_points_list[:-1]#.pop(-1)
    scenario = [map.scenario[-1]]
    road = get_points_from_states(scenario)

       # print("INVALID", invalid)

    return scenario

class CarEnv(Env):
    def __init__(self):
        self.max_number_of_points = 30
        self.action_space = MultiDiscrete([3, cf.model['max_len'] - cf.model['min_len'], cf.model['max_angle'] - cf.model['min_angle']])  # 0 - increase temperature, 1 - decrease temperature
        

        self.map = Map(cf.model['map_size'])
        self.car = Car(cf.model["speed"], cf.model["steer_ang"], cf.model["map_size"])

        self.max_steps = 29
        self.steps = 0


        self.fitness = 0

        self.car_path = []
        self.road = []

        self.t = 0

        self.episode = 0
        self.results = []
        self.all_results = {}
        self.train_episode = 0
        self.evaluation = False
        self.scenario = []
        
        self.observation_space = Box(low=0, high=100, shape = (self.max_number_of_points*3, ), dtype=np.int8)
    def set_state(self, action):
        if action[0] == 0:
            distance  = action[1] + cf.model['min_len']
            self.done = not(self.map.go_straight(distance))
            angle = 0
        elif action[0] == 1:
            angle = action[2] + cf.model['min_angle']
            self.done = not(self.map.turn_right(angle))
            distance = 0
        elif action[0] == 2:
            angle = action[2] + cf.model['min_angle']
            self.done = not(self.map.turn_left(angle))
            distance = 0

        return [action[0], distance, angle]

    def step(self, action):
        assert self.action_space.contains(action)
        self.done = False

        self.state[self.steps] = self.set_state(action)

        if self.done is True:
            reward = 5
        else:


            points = get_points_from_states(self.state[:self.steps])
            intp_points = self.car.interpolate_road(points)

            self.road = intp_points

            new_reward, self.car_path, _ = self.car.execute_road(intp_points)

            self.fitness = new_reward

            reward = new_reward - self.old_fitness

            if reward > 0:
                reward *= 10



            if new_reward < 0:
                reward = -5
                self.done = True
        
            self.old_fitness = self.fitness

            current_state = self.state.copy()
            self.all_fitness.append(self.fitness)
            self.all_states.append(current_state)
        #self.render()
        #print("REWARD", reward)


        self.steps += 1

        if self.steps >= self.max_steps:
            self.done = True
        

        info = {}
        obs = [coordinate for tuple in self.state for coordinate in tuple]


        return np.array(obs, dtype=np.int8), reward, self.done, info


    def evaluate_scenario(self):
        points = get_points_from_states(self.state[:self.steps])
        intp_points = self.car.interpolate_road(points)

        reward, self.car_path, _= self.car.execute_road(intp_points)
        self.fitness = reward
        return reward

    def reset(self):
        #print("Reset")

        self.steps = 1
        #print(self.fitness)
        self.map = Map(cf.model['map_size'])
        self.car = Car(cf.model["speed"], cf.model["steer_ang"], cf.model["map_size"])
        self.state = generate_random_state()#road()
        while len(self.state) < self.max_number_of_points:
            self.state.append([0, 0, 0])

        self.old_fitness = self.evaluate_scenario()

        #print(self.state.shape)
        #print(self.t)
        
        self.road = []
        self.car_path = []

        self.all_states = []
        self.all_fitness = []



        obs = [coordinate for tuple in self.state for coordinate in tuple]
        #print("Observation1: ", len(obs))


        return np.array(obs, dtype=np.int8)

    def render(self, mode='human'):

        #if self.done:
        fig, ax = plt.subplots(figsize=(12, 12))
        road_x = []
        road_y = []
        for p in self.road:
            road_x.append(p[0])
            road_y.append(p[1])

        ax.plot(road_x, road_y, "yo--", label="Road")


        if len(self.car_path):
            ax.plot(self.car_path[0], self.car_path[1], "bo", label="Car path")

        
        top = cf.model["map_size"]
        bottom = 0

        ax.set_title(
            "Test case fitenss " + str(self.fitness) , fontsize=17
        )

        ax.set_ylim(bottom, top)

        ax.set_xlim(bottom, top)
        ax.legend()

        #save_path = os.path.join(cf.files["tc_img"], "generation_" + str(self.episode))
        if os.path.exists(cf.files["img_path"]) == False:
            os.mkdir(cf.files["img_path"])

        fig.savefig(cf.files["img_path"] + str(self.episode) + "_" + str(self.fitness) + ".png")
        fig.savefig("test.png")

        plt.close(fig)


        pass

def calculate_straight(scenario):
    straight = 0
    for item in scenario:
        if item[0] == 0:
            straight += item[1]
    return straight


def compare_states( state1, state2):
    similarity = 0
    if state1[0] == state2[0]:
        similarity += 1
        if state1[0] == 0:
            if abs(state1[1] - state2[1]) <= 10:
                similarity += 1
        else:
            if abs(state1[2] - state2[2]) <= 10:
                similarity += 1

    return similarity

def calc_novelty(old, new):
    similarity = 0
    #print("OLD", old)
    #print("NEW", new)
    total_states = (len(old) + len(new))*2

    if len(old) > len(new):
        for i in range(len(new)):
            similarity += compare_states(old[i], new[i])
    elif len(old) <= len(new):
        for i in range(len(old)):
            similarity += compare_states(old[i], new[i])
    novelty = 1 - (similarity/total_states)
    return novelty



if __name__ == "__main__":

 


    print("Starting...")

    final_results = {}
    final_novelty ={}
    scenario_list = []
    novelty_list = []

    m = 0
    for m in range(3):

        

        environ = CarEnv()
        

        checkpoint_callback_ppo = CheckpointCallback(save_freq=10000, save_path=cf.files["model_path"],
                                                name_prefix='rl_model_new' + str(m))

        log_path = cf.files["logs_path"]

            
        '''
        start = time.time()
        model = PPO('MlpPolicy', environ,  verbose=1,  tensorboard_log=log_path, device='cuda')

        model.learn(total_timesteps=1000000, tb_log_name="ppo", callback=checkpoint_callback_ppo)  

        print("Training time: {}".format(time.time() - start))
        print(log_path)
        # test the environment
        '''

        #mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
        
        
        model_save_path = cf.files['model_path'] + "rl_model_new0_1000000_steps.zip"
        model = PPO.load(model_save_path)
        

        episodes = 30

        environ.evaluation = True
        environ.state = []
        
        i = 0
        results = []
        while environ.episode < episodes:
            obs = environ.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs)

                obs, rewards, done, info = environ.step(action)
            i += 1
            max_fitness = max(environ.all_fitness)
            max_index = np.argmax(environ.all_fitness)
            print(max_fitness)

            if (max_fitness > 7) or i >15:
                print(i)
                print("Round: {}".format(environ.episode))
                print("Max fitness: {}".format(max_fitness))
                
                #scenario = environ.state[:environ.t]
                scenario = environ.all_states[max_index]
                environ.render(scenario)
                scenario_list.append(scenario)
                environ.episode += 1
                results.append(max_fitness)
                i  = 0

        #results = environ.results
        

        final_results[str(m)] = results

        
        novelty_list = []
        for i in combinations(range(0, 30), 2):
            current1 = scenario_list[i[0]]
            current2 = scenario_list[i[1]]
            nov = calc_novelty(current1, current2)
            novelty_list.append(nov)
        novelty = abs(sum(novelty_list)/len(novelty_list))

        final_novelty[str(m)] = novelty

        scenario_list = []

        
        with open('2022-09-21-results-ppo.txt', 'w') as f:
            json.dump(final_results, f, indent=4)

        with open('2022-09-21-novelty-ppo.txt', 'w') as f:
            json.dump(final_novelty, f, indent=4)
        
        




