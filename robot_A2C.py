'''
https://github.com/FitMachineLearning/FitML/
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
Article about this solution
https://github.com/FitMachineLearning/FitML/edit/master/ActorCritic/README.md
Using Actor Critic
Note that I prefer the terms Action Predictor Network and Q/Reward Predictor network better
Update
Cleaned up variables and more readable memory
Improved hyper parameters for better performance
'''
from simulation_utils import box, simulation
from kinematics import pose3D
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

num_env_variables = 30
num_env_actions = 5
num_initial_observation = 50
learning_rate = 0.001
apLearning_rate = 0.001
weigths_filename = "obstacle-avoidance-v1-weights.h5"
apWeights_filename = "obstacle-avoidance-v1-weights-ap.h5"

# range within wich the SmartCrossEntropy action parameters will deviate from
# remembered optimal policy
sce_range = 0.2
b_discount = 0.80
max_memory_len = 50000
starting_explore_prob = 0.4
training_epochs = 4
mini_batch = 256
load_previous_weights = False
observe_and_train = True
save_weights = True
num_games_to_play = 30000

# One hot encoding array
possible_actions = np.arange(0, num_env_actions)
actions_1_hot = np.zeros((num_env_actions, num_env_actions))
actions_1_hot[np.arange(num_env_actions), possible_actions] = 1

# Create testing enviroment
box1 = box([6, 20, 15], pos=[-3, 10, 0])
box2 = box([10, 10, 40], pos=[-5, 20, 0])
obstacles = np.array([box1,box2])



position = np.array([-15, 25, 10])
initial_pose = pose3D(position, False)
position = np.array([15, 25, 10])
target_pose = pose3D(position, False)
cut_off = 3
radius = 4.0

env = simulation(initial_pose, target_pose, obstacles, radius=radius, cut_off=cut_off)

env.reset()

# nitialize the Reward predictor model
Qmodel = Sequential()
# model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
Qmodel.add(Dense(256, activation='relu', input_dim=(num_env_variables + num_env_actions)))
Qmodel.add(Dropout(0.5))
Qmodel.add(Dense(256, activation='relu'))
Qmodel.add(Dropout(0.5))
Qmodel.add(Dense(256, activation='relu'))
Qmodel.add(Dropout(0.2))
Qmodel.add(Dense(1))
opt = optimizers.adam(lr=learning_rate)
Qmodel.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

# initialize the action predictor model
action_predictor_model = Sequential()
# model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
action_predictor_model.add(Dense(256, activation='relu', input_dim=num_env_variables))
action_predictor_model.add(Dropout(0.5))
action_predictor_model.add(Dense(256, activation='relu'))
action_predictor_model.add(Dropout(0.5))
action_predictor_model.add(Dense(256, activation='relu'))
action_predictor_model.add(Dropout(0.2))
action_predictor_model.add(Dense(num_env_actions))

opt2 = optimizers.adam(lr=apLearning_rate)
action_predictor_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])

# load previous model weights if they exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/" + weigths_filename
    print("filepath ", fn)
    if os.path.isfile(fn):
        print("loading weights")
        Qmodel.load_weights(weigths_filename)
    else:
        print("File ", weigths_filename, " does not exis. Retraining... ")

# load previous action predictor model weights if they exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/" + apWeights_filename
    print("filepath ", fn)
    if os.path.isfile(fn):
        print("loading weights")
        action_predictor_model.load_weights(apWeights_filename)
    else:
        print("File ", apWeights_filename, " does not exis. Retraining... ")

memorySA = np.zeros(shape=(1, num_env_variables + num_env_actions))
memoryS = np.zeros(shape=(1, num_env_variables))
memoryA = np.zeros(shape=(1, 1))
memoryR = np.zeros(shape=(1, 1))

mstats = []


def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate, action), axis=0)
    predX = np.zeros(shape=(1, num_env_variables + num_env_actions))
    predX[0] = qs_a

    # print("trying to predict reward at qs_a", predX[0])
    pred = Qmodel.predict(predX[0].reshape(1, predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward


def GetRememberedOptimalPolicy(qstate):
    predX = np.zeros(shape=(1, num_env_variables))
    predX[0] = qstate

    # print("trying to predict reward at qs_a", predX[0])
    pred = action_predictor_model.predict(predX[0].reshape(1, predX.shape[1]))
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy


if observe_and_train:

    # Play the game 500 times
    for game in range(num_games_to_play):
        gameSA = np.zeros(shape=(1, num_env_variables + num_env_actions))
        gameS = np.zeros(shape=(1, num_env_variables))
        gameA = np.zeros(shape=(1, num_env_actions))
        gameR = np.zeros(shape=(1, 1))
        total_reward = 0
        # Get the Q state
        qs = env.reset()
        # print("qs ", qs)
        if game < num_initial_observation:
            print("Observing game ", game)
        else:
            print("Learning & playing game ", game)
        for step in range(5000):

            if game < num_initial_observation and not load_previous_weights:
                # take a random action
                a = env.random_action()
                
            else:
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob - (starting_explore_prob / num_games_to_play) * game

                # Chose between prediction and chance
                if prob < explore_prob:
                    # take a random action
                    a = env.random_action()

                else:

                    # Get Remembered optiomal policy
                    remembered_optimal_policy = GetRememberedOptimalPolicy(qs)

                    stock = np.zeros(9)
                    stockAction = np.zeros(shape=(9, num_env_actions))
                    for i in range(9):
                        stockAction[i] = env.random_action()
                        stock[i] = predictTotalRewards(qs, stockAction[i])
                    best_index = np.argmax(stock)
                    randaction = stockAction[best_index]

                    # Compare R for SmartCrossEntropy action with remembered_optimal_policy and select the best
                    # if predictTotalRewards(qs,remembered_optimal_policy) > utility_possible_actions[best_sce_i]:
                    if predictTotalRewards(qs, remembered_optimal_policy) > predictTotalRewards(qs, randaction):
                        a = remembered_optimal_policy
                        # print(" | selecting remembered_optimal_policy ",a)
                    else:
                        a = randaction
                        # print(" - selecting generated optimal policy ",a)

            qs_a = np.concatenate((qs, a), axis=0)

            # get the target state and reward
            s, r, done, info = env.step(a)
            # record only the first x number of states
            total_reward += r
            
            if step == 0:
                gameSA[0] = qs_a
                gameS[0] = qs
                gameR[0] = np.array([r])
                gameA[0] = np.array([r])
            else:
                gameSA = np.vstack((gameSA, qs_a))
                gameS = np.vstack((gameS, qs))
                gameR = np.vstack((gameR, np.array([r])))
                gameA = np.vstack((gameA, np.array([a])))

            if done:
                # Calculate Q values from end to start of game
                mstats.append(step)
                for i in range(0, gameR.shape[0]):
                    # print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i == 0:
                        # print("reward at the last step ",gameY[(gameY.shape[0]-1)-i][0])
                        gameR[(gameR.shape[0] - 1) - i][0] = gameR[(gameR.shape[0] - 1) - i][0]
                    else:
                        # print("local error before Bellman", gameY[(gameY.shape[0]-1)-i][0],"Next error ", gameY[(gameY.shape[0]-1)-i+1][0])
                        gameR[(gameR.shape[0] - 1) - i][0] = gameR[(gameR.shape[0] - 1) - i][0] + b_discount * \
                                                             gameR[(gameR.shape[0] - 1) - i + 1][0]
                        # print("reward at step",i,"away from the end is",gameY[(gameY.shape[0]-1)-i][0])
                    if i == gameR.shape[0] - 1:
                        print("Training Game #", game, "memory ", memoryR.shape[0], " steps = ", step, "last reward", r,
                              " finished with score ", total_reward)
                    
                total_reward = 0
                if memoryR.shape[0] == 1:
                    memorySA = gameSA
                    memoryR = gameR
                    memoryA = gameA
                    memoryS = gameS
                else:
                    # Add experience to memory
                    memorySA = np.concatenate((memorySA, gameSA), axis=0)
                    memoryS = np.concatenate((memoryS, gameS), axis=0)
                    memoryR = np.concatenate((memoryR, gameR), axis=0)
                    memoryA = np.concatenate((memoryA, gameA), axis=0)

                # if memory is full remove first element
                if np.alen(memorySA) >= max_memory_len:
                    memorySA = memorySA[gameR.shape[0]:]
                    memoryR = memoryR[gameR.shape[0]:]
                    memoryA = memoryA[gameR.shape[0]:]
                    memoryS = memoryS[gameR.shape[0]:]
                    
            # Update the states
            qs = s

            # Retrain every X failures after num_initial_observation
            if done and game >= num_initial_observation:
                if game % 5 == 0:
                    print("Training  game# ", game, "memory size", memorySA.shape[0])

                    # training Reward predictor model
                    Qmodel.fit(memorySA, memoryR, batch_size=mini_batch, epochs=training_epochs, verbose=0)

                    # training action predictor model
                    action_predictor_model.fit(memoryS, memoryA, batch_size=mini_batch, epochs=training_epochs,
                                               verbose=0)

            if done and game >= num_initial_observation:
                if save_weights and game % 20 == 0:
                    # Save model
                    print("------------------Saving weights-------------------")
                    name_actor = "./save/actor_episode_" + str(game) + "_score_ " + str(total_reward) + "_.h5"
                    name_critic = "./save/critic_episode_" + str(game) + "_score_ " + str(total_reward) + "_.h5"
            
                    Qmodel.save_weights(name_actor)
                    action_predictor_model.save_weights(name_critic)

            if done:
                '''
                #Game won  conditions
                if step > 197:
                    print("Game ", game," WON *** " )
                else:
                    print("Game ",game," ended with positive reward ")
                #Game ended - Break
                '''
                break

plt.plot(mstats)
plt.show()

if save_weights:
    # Save model
    print("------------------Saving weights-------------------")
    Qmodel.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)
