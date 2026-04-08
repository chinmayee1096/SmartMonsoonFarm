from fastapi import FastAPI
from env.environment import MonsoonFarmEnv

app = FastAPI()
env = MonsoonFarmEnv()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {"observation": env.reset()}

@app.post("/step")
def step(action: dict):
    obs, reward, done, info = env.step(action.get("action", None))
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()