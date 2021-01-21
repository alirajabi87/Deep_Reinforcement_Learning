import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from collections import deque
import time
import numpy as np
import random

import cv2 as cv
from PIL import Image
from tqdm import tqdm
import shutil
import pandas as pd
import matplotlib.pyplot as plt

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "TF"
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200  # For model save

# Envroment Settings
EPISODES = 20_000

# exploration settings
epsilon = 0.8
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes ?
SHOW_PREVIEW = False

if os.path.exists('../logs'):
    shutil.rmtree('../logs', ignore_errors=True)


class ModifiedTensorBoard(TensorBoard):

    # We need one log file for all fit() calls
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)  # FileWrite

    # over ride this function to prevent creating default log writer
    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def _get_log_write_dir(self):
        pass

    def update_stats(self, stats):
        self._write_logs(stats)

    def _write_logs(self, logs):
        for name, value in logs.items():
            tf.summary.scalar(name, value, step=self.step)
        self.writer.flush()


class DQNAgent(Model):

    def __init__(self, env):
        super().__init__()
        # main model # get trained every step
        self.model = self.create_model(env)

        # Target model # get predict every step
        self.target_model = self.create_model(env)
        self.target_model.set_weights(weights=self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # it is used for sampling from 50000 to make batches
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/Modified/{MODEL_NAME}-{int(time.time())}")
        self.tensorboard1 = TensorBoard(log_dir="../logs/train\\")
        self.target_update_counter = 0

    def create_model(self, env):
        if os.path.isfile('../models/temp.h5'):
            model = tf.keras.models.load_model('../models/temp.h5')
            epsilon = 0.1
            print("Model is loaded")
        else:
            print("Creating Model from Model.model")
            i = Input(env.OBSERVATION_SPACE_VALUES)
            x = Conv2D(256, (3, 3))(i)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2))(x)

            x = Conv2D(256, (3, 3))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2))(x)

            x = Flatten()(x)
            x = Dense(128)(x)
            x = Activation('relu')(x)
            x = Dropout(0.4)(x)
            x = Dense(64)(x)
            x = Activation('relu')(x)
            x = Dropout(0.4)(x)
            prediction = Dense(env.ACTION_SPACE_SIZE, activation='linear')(x)  # linear softmax

            model = Model(inputs=i, outputs=prediction)
            print(model.summary())
            model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255.)[0]

    def train(self, terminal_state, step):
        batchSize = 32
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            # print(f"in the if statement in train {len(self.replay_memory)}")
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255.
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255.
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # update qs
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)
        # print(self.model.summary())
        self.model.fit(np.array(X) / 255, np.array(Y),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       callbacks=[self.tensorboard1] if terminal_state else None,
                       shuffle=False, )

        # Updating to determine if we want to update target_model yet?
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights((self.model.get_weights()))
            self.target_update_counter = 0


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=0, y=1)
        elif choice == 6:
            self.move(x=-1, y=0)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1

        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1


class BlobEnv:
    SIZE = 10
    RETURN_IMAGE = True

    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25

    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9

    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3

    d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}  # BGR

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)

        while self.food == self.player:
            self.food = Blob(self.SIZE)

        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episod_step = 0

        if self.RETURN_IMAGE:
            observation = np.array(self.get_Image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)

        return observation

    def step(self, action):
        self.episod_step += 1
        self.player.action(action)

        # self.enemy.move()
        # self.enemy.move()

        if self.RETURN_IMAGE:
            new_observation = np.array(self.get_Image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.ENEMY_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episod_step >= 200:
            done = True

        return new_observation, reward, done

    def get_Image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]

        img = Image.fromarray(env, 'RGB')
        return img

    def render(self):
        img = self.get_Image()
        img = img.resize((300, 300))
        cv.imshow("image", np.array(img))
        cv.waitKey(1)


env = BlobEnv()
agent = DQNAgent(env=env)

ep_rewards = [-200]
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

if not os.path.exists('../models'):
    os.mkdir('../models')

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1

    current_state = env.reset()

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action=action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % 1:  # AGGREGATE_STATS_EVERY
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1

    ep_rewards.append(episode_reward)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[:AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        mydic = dict(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        # print(mydic)
        agent.tensorboard.update_stats(mydic)

        if average_reward >= MIN_REWARD:
            print(f"Avg_reward: {average_reward}")
            print(f"MIN REWARD: {MIN_REWARD}")
            agent.model.save(
                f"../models/{MODEL_NAME}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}min_{int(time.time())}.h5")
        else:
            # print(f"Avg_reward: {average_reward}")
            # print(f"MIN REWARD: {MIN_REWARD}")
            agent.model.save('../models/temp.h5')

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(epsilon, MIN_EPSILON)
