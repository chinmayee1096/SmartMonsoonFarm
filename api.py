from fastapi import FastAPI
from env.environment import MonsoonFarmEnv

app = FastAPI()

env = MonsoonFarmEnv()

@app.get("/")
def home():
    return {"message": "API running"}   # 👈 ADD THIS

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": str(state)}

@app.post("/step")
def step(action: dict):
    action_value = action.get("action", 0)
    next_state, reward, done, info = env.step(action_value)
    return {
        "state": str(next_state),
        "reward": reward,
        "done": done,
        "info": info
    }