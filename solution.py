import sys
import time
from constants import *
from environment import *
from state import State
import numpy as np
"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""


class RLAgent:

    #
    # TODO: (optional) Define any constants you require here.
    #

    def __init__(self, environment: Environment):
        self.environment = environment
        self.q_values = {}   
                 
        #
        # TODO: (optional) Define any class instance variables you require (e.g. Q-value tables) here.
        #
        pass

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        #
        # TODO: Implement your Q-learning training loop here.
        #

        # List for keeping track of epsiode rewards 
        episode_reward_list = []
        r100 = float('-inf')

        # Starting timer for time termination condition
        start = time.time()
        ########### repeat for each epsiode
        while r100 < self.environment.evaluation_reward_tgt and (time.time()-start)<self.environment.training_time_tgt:
            #print('inside r100 loop')
            steps = 0
            episode_reward = 0

            # Initialize s
            init_state = self.environment.get_init_state()
            current_state = State(self.environment, init_state.robot_posit, init_state.robot_orient)
            # Repeat (for each step of the epsiode)
            #################### BEGIN EPSIODE ############################
            while not self.environment.is_solved(current_state) and steps < 100: 
                #print('inside episode loop')
                steps += 1

                reward, next_state = self.next_iteration(current_state)
                current_state = next_state
                episode_reward += reward
            ############### END EPISODE ##################################
            episode_reward_list.append(episode_reward) 
            r100 = np.mean(episode_reward_list[-100:])
            print('r100: ', r100, 'len r100: ', len(episode_reward_list))
        
    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  Q-learning Q-values) here.
        #

        return self.get_best_action(state)
        

    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """

        # self.q_learn_train()
        #
        # TODO: Implement your SARSA training loop here.
        #

        episode_reward_list = []
        r100 = float('-inf')

        # Starting timer for time termination condition
        start = time.time()
        ########### repeat for each epsiode
        while r100 < self.environment.evaluation_reward_tgt and (time.time()-start)<self.environment.training_time_tgt:
            #print('inside r100 loop')
            steps = 0
            episode_reward = 0

            # Initialize s
            init_state = self.environment.get_init_state()
            current_state = State(self.environment, init_state.robot_posit, init_state.robot_orient)
            # Choose action from state using epsilon greedy
            action = self.choose_action(current_state)

            # Repeat (for each step of the epsiode)
            #################### BEGIN EPSIODE ############################
            while not self.environment.is_solved(current_state) and steps < 100: 
                #print('inside episode loop')
                steps += 1

                reward, next_state, next_action = self.next_iteration_SARSA(current_state, action)
                current_state = next_state
                action = next_action
                episode_reward += reward
            ############### END EPISODE ##################################
            episode_reward_list.append(episode_reward) 
            r100 = np.mean(episode_reward_list[-100:])
            print('r100: ', r100, 'len r100: ', len(episode_reward_list))
        pass

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  SARSA Q-values) here.
        #
        return self.get_best_action(state)

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: (optional) Add any additional methods here.
    #
    #

    def next_iteration(self, state: State):
        """
        Write a method to update your agent's q_values here
        Include steps to generate new state-action q_values as you go
        """

        # Choose action from state using epsilon greedy
        action = self.choose_action(state)
        # Take action, observe reward and next state
        reward, next_state = self.environment._Environment__apply_dynamics(state, action)

        # Update q-value for the (state, action) pair
        # If it's a new state action pair, give it q_val 0
        old_q = self.q_values.get((state, action), 0)
        best_next = self.get_best_action(next_state)

        best_next_q = self.q_values.get((next_state, best_next), 0)
        if self.environment.is_solved(next_state):
            best_next_q = 0
        target = reward + self.environment.gamma * best_next_q

        new_q = old_q + self.environment.alpha * (target - old_q)
        self.q_values[(state, action)] = new_q

        return reward, next_state

    def get_best_action(self, state: State):
        best_q = float('-inf')
        best_a = None
        for action in ROBOT_ACTIONS:
            this_q = self.q_values.get((state, action))
            if this_q is not None and this_q > best_q:
                best_q = this_q
                best_a = action
        return best_a

    def choose_action(self, state: State):
        # Using epsilon-greedy
        EPSILON = 0.2 # 'epsilon' in epsilon greedy
        exploit_prob = 1-EPSILON
        best_a = self.get_best_action(state)
        if best_a is None or random.random() < exploit_prob:
            return random.choice(ROBOT_ACTIONS)
        return best_a

    def next_iteration_SARSA(self, state: State, action):
        """
        Write a method to update your agent's q_values here
        Include steps to generate new state-action q_values as you go
        """
        # Take action, observe reward and next state
        reward, next_state = self.environment._Environment__apply_dynamics(state, action)

        # Update q-value for the (state, action) pair
        # If it's a new state action pair, give it q_val 0
        old_q = self.q_values.get((state, action), 0)
        next_action = self.choose_action(next_state)

        best_next_q = self.q_values.get((next_state, next_action), 0)
        if self.environment.is_solved(next_state):
            best_next_q = 0
        target = reward + self.environment.gamma * best_next_q

        new_q = old_q + self.environment.alpha * (target - old_q)
        self.q_values[(state, action)] = new_q

        return reward, next_state, next_action
        
