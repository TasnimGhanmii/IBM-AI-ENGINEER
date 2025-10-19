import gym, numpy as np, tensorflow as tf, random, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

warnings.filterwarnings("ignore")

# --- environment ---
env = gym.make("CartPole-v1")
np.random.seed(42); env.reset(seed=42)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- DQN ---
def build_model(state_size, action_size):
    model = Sequential([
        Dense(24, input_dim=state_size, activation="relu"),
        Dense(24, activation="relu"),
        Dense(action_size, activation="linear")
    ])
    model.compile(loss="mse", optimizer=Adam(1e-3))
    return model

model = build_model(state_size, action_size)

# --- hyper-parameters ---
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95
memory = deque(maxlen=2000)
batch_size = 32
episodes = 50

# --- helpers ---
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

# --- training loop ---
for e in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = np.reshape(state, [1, state_size])
    score = 0
    for t in range(200):
        action = act(state)
        result = env.step(action)
        next_state, reward, done = result[:3]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state
        score += 1
        if done:
            break
        if len(memory) > batch_size:
            replay(batch_size)
    print(f"Episode {e+1}/{episodes}  score {score}  ε {epsilon:.3f}")

# --- evaluation ---
print("\nEvaluation (10 episodes)")
scores = []
for e in range(10):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = np.reshape(state, [1, state_size])
    total = 0
    for t in range(200):
        action = np.argmax(model.predict(state, verbose=0)[0])
        state, reward, done = env.step(action)[:3]
        if isinstance(state, tuple):
            state = state[0]
        state = np.reshape(state, [1, state_size])
        total += reward
        if done:
            break
    scores.append(total)
    print(f"Episode {e+1}/10  score {total}")

print(f"Average score: {np.mean(scores):.1f}")
env.close()