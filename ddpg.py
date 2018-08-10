import random
import numpy as np
from gym_torcs import TorcsEnv

import tensorflow as tf
from keras import backend as K

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer

from OU import OU
import timeit
import json
import os

from time import time
from datetime import datetime

import threading

save_folder = "training_data/"

# OU Noise generating function
def OUNoise(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)

def run( train_flag=0, run_ep_count=1, current_run=0):
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = .99
    TAU = .001
    LRA = .0001
    LRC = .001
    hidden1 = 500
    hidden2 = 1000

    ## TODO: Make this part dynamic maybe, parametrize with observations
    state_dim = 65
    action_dim = 2

    #Men of culture
    np.random.seed( 1337)

    # Torcs config
    vision = False

    EXPLORE = 100000.
    max_steps = 100000
    episode_count = run_ep_count

    #Gym Env related
    ep_reward = 0
    step_count = 0
    done = False
    epsilon = 1
    step = 0

    lap_limiter = 2 #Evaluation
    if train_flag == 1: lap_limiter = 4 #Training on 3 laps

    # Score trace
    scores = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess =tf.Session( config=config)

    K.set_session( sess)

    actor = ActorNetwork( sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA,
        hidden1, hidden2)
    critic = CriticNetwork( sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC,
        hidden1, hidden2)
    buffer = ReplayBuffer( BUFFER_SIZE)

    #Agent only
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + "/raceconfig/agent_practice.xml"

    env = TorcsEnv(vision=vision, throttle=True,gear_change=False,
		race_config_path=race_config_path, rendering=False,
        lap_limiter=lap_limiter)

    print( "Attempting to load weights")
    try:
        actor.model.load_weights( save_folder + "run_" + str( current_run) + "_actormodel.h5")
        critic.model.load_weights( save_folder + "run_" + str( current_run) + "_criticmodel.h5")
        actor.target_model.load_weights( save_folder + "run_" + str( current_run) + "_actormodel.h5")
        critic.target_model.load_weights( save_folder + "run_" + str( current_run) + "_criticmodel.h5")
        print( "Weights loaded")
    except Exception as ex:
        print( "Failed to load the weights:")
        print( ex)

    for i in range(episode_count):
        print("Run " + str( current_run) + " - Episode : " + str(i) + " Replay Buffer " + str( buffer.size()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        #Stats
        total_reward = 0.

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,
            ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents/200.))

        for j in range(max_steps):

            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            #Adding noise to action during training ?
            noise_t[0][0] = train_flag * max(epsilon, 0) * OUNoise(x=a_t_original[0][0],  mu=0.0 , theta=0.60, sigma=0.30)
            noise_t[0][1] = train_flag * max(epsilon, 0) * OUNoise(x=a_t_original[0][1],  mu=0.5 , theta=1.00, sigma=0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX,
                ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents/200.))

            buffer.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer

            batch = buffer.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if train_flag:
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            if step % 100 == 0:
                print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

            if np.mod(i, 3) == 0:

                if (train_flag):
                    print("Now we save model")
                    actor.model.save_weights(save_folder + "run_" + str( current_run) +"_actormodel.h5", overwrite=True)
                    with open(save_folder + "run_" + str( current_run) + "_actormodel.json", "w") as outfile:
                        json.dump(actor.model.to_json(), outfile)

                    critic.model.save_weights(save_folder + "run_" + str( current_run) + "_criticmodel.h5", overwrite=True)
                    with open(save_folder + "run_" + str( current_run) + "_criticmodel.json", "w") as outfile:
                        json.dump(critic.model.to_json(), outfile)

            print("Run " + str( current_run) +" - Episode " + str( i) + ": Return " + str(total_reward))
            print("Total Step: " + str(step))
            # print("")

            scores.append( total_reward)

            # Dump scores in case of unplanned interrupt
            if step % 100 == 0 and train_flag == 1:
                with open( save_folder + "run_" + str( current_run) + "_scores.json", "w") as outfile:
                    json.dump( scores, outfile)
            if step % 100 == 0 and train_flag == 0:
                with open( save_folder + "run_" + str( current_run) + "_eval_scores.json", "w") as outfile:
                    json.dump( scores, outfile)

    env.end()  # This is for shutting down TORCS
    print("Run finish.")

    return scores

def data_dumping( i_run, train_scores, eval_scores, startDateTimeStr):
    # Dump scores in case of unplanned interrupt
    with open( save_folder + "run_" + str( i_run) + "_scores.json", "w") as outfile:
            json.dump( train_scores[i_run], outfile)

    # Dump scores in case of unplanned interrupt
    with open( save_folder + "run_" + str( i_run) + "_eval_scores.json", "w") as outfile:
            json.dump( eval_scores[i_run], outfile)

    #Dump after each training
    print("Fusing training and eval data\n")
    full_data = { "train_scores": train_scores,
        "eval_scores": eval_scores}

    try:
        filename = save_folder + "dist_only_@{}_full.json".format(
            startDateTimeStr)

        print( "Writing full data to \"" + filename + "\"\n")
        with open( filename, "w") as outfile:
            json.dump( full_data, outfile)
    except Exception as ex:
        print( "Saving file:\n")
        print( ex)

    print( "Threaded data dumpiung comnplete\n")

## TODO: Parametrize with respect to training, episode counts, etc
## find a parser maybe
if __name__ == "__main__":
    startDateTimeStr = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]

    train_count = 5
    train_ep_count = 3000

    eval_ep_count = 10

    train_scores = [] # train_scores
    eval_scores = []

    for i_run in range( train_count):
        train_score = run( train_flag=1,
            run_ep_count = train_ep_count, current_run = i_run)

        #Saves score trace for all episodes in the rn
        train_scores.append( train_score)

        eval_scores.append( run( train_flag = 0,
            run_ep_count=eval_ep_count, current_run= i_run))

        # TODO: Using threading to dump data so training doesn't need to wait
        threading.Thread( target=data_dumping( i_run, train_scores,
            eval_scores, startDateTimeStr)).start()
