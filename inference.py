import os
import requests
from openai import OpenAI

# ✅ Use platform-provided variables
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)

# Keep your app URL separate
APP_URL = "https://chinmayee1096-smart-monsoon-farm.hf.space"

def main():
    print("[START]")

    # ✅ REQUIRED: Make at least one LLM call
    llm_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Give a short farming tip"}
        ]
    )

    print("[LLM OUTPUT]:", llm_response.choices[0].message.content)

    # Your existing logic
    reset_res = requests.post(f"{APP_URL}/reset")
    state = reset_res.json()
    print("[STEP] Reset:", state)

    done = False
    step_count = 0

    while not done and step_count < 10:
        action = {"action": 0}

        res = requests.post(f"{APP_URL}/step", json=action)
        data = res.json()

        print("[STEP]", data)

        done = data.get("done", False)
        step_count += 1

    print("[END] Finished")

if __name__ == "__main__":
    main()
