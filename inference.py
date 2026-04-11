import os
import requests
from openai import OpenAI

def main():
    print("[START]")

    # ✅ Initialize client safely
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )

        # ✅ Safe LLM call
        try:
            response = client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
                messages=[
                    {"role": "user", "content": "Give a short farming tip"}
                ],
                max_tokens=50
            )

            print("[STEP] LLM Response:", response.choices[0].message.content)

        except Exception as e:
            print("[STEP] LLM call failed but continuing:", str(e))

    except Exception as e:
        print("[STEP] Client init failed:", str(e))

    # ✅ Your environment logic
    APP_URL = "https://chinmayee1096-smart-monsoon-farm.hf.space"

    try:
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

    except Exception as e:
        print("[STEP] Environment error:", str(e))

    print("[END] Finished")


if __name__ == "__main__":
    main()
