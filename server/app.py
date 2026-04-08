from fastapi import FastAPI
from env.environment import MonsoonFarmEnv

app = FastAPI()
env = MonsoonFarmEnv()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}


@app.post("/step")
def step(action: dict):
    result = env.step(action)
    return result


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()