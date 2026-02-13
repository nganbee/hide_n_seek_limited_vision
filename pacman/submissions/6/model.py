import numpy as np
import random
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================

GRID = 21
STATE_SIZE = GRID * GRID      # 🔑 BẮT BUỘC 441
ACTION_SIZE = 5

GAMMA = 0.9
LR = 1e-4
EPOCHS = 3000
BATCH_SIZE = 64

MAX_Q = 30.0
GRAD_CLIP = 5.0

MODEL_DIR = Path(__file__).parent / "ml"
MODEL_DIR.mkdir(exist_ok=True)


# ======================================================
# MODEL
# ======================================================

def he_init(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)


def init_model():
    return {
        "W1": he_init(STATE_SIZE, 256),
        "b1": np.zeros(256),
        "W2": he_init(256, 128),
        "b2": np.zeros(128),
        "W3": he_init(128, ACTION_SIZE),
        "b3": np.zeros(ACTION_SIZE),
    }


def relu(x):
    return np.maximum(0, x)


def forward(model, x):
    z1 = x @ model["W1"] + model["b1"]
    a1 = relu(z1)
    z2 = a1 @ model["W2"] + model["b2"]
    a2 = relu(z2)
    q = a2 @ model["W3"] + model["b3"]
    return q, (x, z1, a1, z2, a2)


# ======================================================
# ENV SIMULATION (MATCH AGENT FORMAT)
# ======================================================

def random_map():
    """
    Encoding:
     0 : empty
     1 : wall
     2 : pacman
    -1 : ghost
    """
    m = np.random.choice([0, 1], size=(GRID, GRID), p=[0.85, 0.15])
    return m


def random_empty(m):
    while True:
        p = (random.randint(0, GRID - 1), random.randint(0, GRID - 1))
        if m[p] == 0:
            return p


def build_state(m, pac, ghost):
    s = m.copy()
    s[pac] = 2
    s[ghost] = -1
    return (s / 2.0).flatten()   # 🔑 normalize → [-0.5, 1]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step(pos, a, m):
    moves = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
    dx, dy = moves[a]
    nx, ny = pos[0] + dx, pos[1] + dy
    if 0 <= nx < GRID and 0 <= ny < GRID and m[nx, ny] != 1:
        return (nx, ny)
    return pos


# ======================================================
# TRAINING
# ======================================================

model = init_model()
memory = []

# -------- collect experience --------
for _ in range(8000):
    m = random_map()
    pac = random_empty(m)
    ghost = random_empty(m)

    s = build_state(m, pac, ghost)
    a = random.randint(0, ACTION_SIZE - 1)

    pac2 = step(pac, a, m)
    s2 = build_state(m, pac2, ghost)

    d1 = manhattan(pac, ghost)
    d2 = manhattan(pac2, ghost)

    r = d1 - d2
    if d2 == 0:
        r += 10

    memory.append((s, a, r, s2))

print("Experience:", len(memory))


# -------- train loop --------
for epoch in range(EPOCHS):
    batch = random.sample(memory, BATCH_SIZE)

    for s, a, r, s2 in batch:
        q, cache = forward(model, s)
        q2, _ = forward(model, s2)

        target = q.copy()
        target[a] = np.clip(r + GAMMA * np.max(q2), -MAX_Q, MAX_Q)

        error = np.clip(q - target, -GRAD_CLIP, GRAD_CLIP)

        x, z1, a1, z2, a2 = cache

        dW3 = np.outer(a2, error)
        db3 = error

        da2 = model["W3"] @ error
        dz2 = da2 * (z2 > 0)

        dW2 = np.outer(a1, dz2)
        db2 = dz2

        da1 = model["W2"] @ dz2
        dz1 = da1 * (z1 > 0)

        dW1 = np.outer(x, dz1)
        db1 = dz1

        model["W3"] -= LR * dW3
        model["b3"] -= LR * db3
        model["W2"] -= LR * dW2
        model["b2"] -= LR * db2
        model["W1"] -= LR * dW1
        model["b1"] -= LR * db1

    if epoch % 300 == 0:
        print("Epoch", epoch)


# ======================================================
# SAVE
# ======================================================

np.savez(MODEL_DIR / "pacman_model.npz", **model)
np.savez(MODEL_DIR / "ghost_model.npz", **model)

print("✅ MODEL KHỚP AGENT – READY TO RUN")
