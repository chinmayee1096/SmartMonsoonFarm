import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://chinmayee1096-smart-monsoon-farm.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy-model")
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    print("[START]")

    # Reset environment
    reset_res = requests.post(f"{API_BASE_URL}/reset")
    state = reset_res.json()
    print("[STEP] Reset:", state)

    done = False
    step_count = 0

    while not done and step_count < 10:
        action = {"action": 0}  # simple dummy action

        res = requests.post(f"{API_BASE_URL}/step", json=action)
        data = res.json()

        print("[STEP]", data)

        done = data.get("done", False)
        step_count += 1

    print("[END] Finished")

if __name__ == "__main__":
    main()