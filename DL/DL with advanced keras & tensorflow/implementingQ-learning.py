
import gym, numpy as np, tensorflow as tf, random, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque

warnings.filterwarnings("ignore")

# 2.  CartPole environment
env = gym.make("CartPole-v1")
np.random.seed(42); env.action_space.seed(42); env.observation_space.seed(42)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 3.  Q-network
def build_model(state_size, action_size):
    model = Sequential([
        Input(shape=(state_size,)),
        Dense(24, activation="relu"),
        Dense(24, activation="relu"),
        Dense(action_size, activation="linear")
    ])
    model.compile(loss="mse", optimizer=Adam(1e-3))
    return model

model = build_model(state_size, action_size)

# 4.  Hyper-parameters
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
gamma = 0.95
memory = deque(maxlen=2000)
batch_size = 64
episodes = 50
train_freq = 5

# 5.  Helper functions
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    return np.argmax(model.predict(state, verbose=0)[0])

def replay(batch_size):
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)

    states = np.vstack([x[0] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_states = np.vstack([x[3] for x in minibatch])
    dones = np.array([x[4] for x in minibatch])

    q_next = model.predict(next_states, verbose=0)
    q_target = model.predict(states, verbose=0)

    for i in range(batch_size):
        target = rewards[i]
        if not dones[i]:
            target += gamma * np.amax(q_next[i])
        q_target[i][actions[i]] = target

    model.fit(states, q_target, epochs=1, verbose=0)

    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# 6.  Training loop
for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0
    for t in range(500):
        action = act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state
        score += 1
        if done:
            break
        if t % train_freq == 0:
            replay(batch_size)
    print(f"episode {e+1}/{episodes}  score {score}  ε {epsilon:.3f}")

# 7.  Quick visual evaluation (10 episodes, no training)
print("\nEvaluation (10 episodes):")
for e in range(10):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    for t in range(500):
        action = np.argmax(model.predict(state, verbose=0)[0])
        state, _, terminated, truncated, _ = env.step(action)
        state = np.reshape(state, [1, state_size])
        if terminated or truncated:
            print(f"episode {e+1}/10  score {t}")
            break
env.close()