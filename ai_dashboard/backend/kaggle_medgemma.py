"""
Kaggle Notebook – MedGemma 4B Inference Server for Tremor Clinical AI
=====================================================================
Copy each cell (marked by # %% [Cell N]) into a separate Kaggle notebook cell.

Prerequisites:
  1. Go to kaggle.com/models/google/medgemma → Request Access & accept license
  2. Create a new Kaggle Notebook with GPU T4 x2 accelerator
  3. Add MedGemma as a notebook Input (Add Input → Models → google/medgemma)
  4. Add your HuggingFace token as a Kaggle Secret named "HF_TOKEN"
  5. Add your ngrok auth token as a Kaggle Secret named "NGROK_AUTH_TOKEN"
     (get one free at https://dashboard.ngrok.com/signup)
"""

# %% [Cell 1] — Install Dependencies
# ====================================
# !pip install flask pyngrok transformers accelerate -q
 
# %% [Cell 2] — Load MedGemma Model
# ====================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/medgemma-4b-it"

print("⏳ Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("⏳ Loading model (this takes 2-3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("✅ Model loaded!")


# %% [Cell 3] — Define Inference Function
# ==========================================
def generate_response(system_prompt: str, user_prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Build a chat-style prompt, run MedGemma, return the generated text.
    """
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
        )

    # Decode only the NEW tokens (skip the input prompt tokens)
    generated = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )
    return generated


# %% [Cell 4] — Quick Test (Optional)
# ======================================
test_result = generate_response(
    system_prompt=(
        "You are a clinical reasoning assistant specialized in tremor "
        "pattern interpretation. You do not diagnose disease."
    ),
    user_prompt=(
        "Analyze a tremor session:\n"
        "- Dominant band: 4-6 Hz (resting)\n"
        "- Mean tremor score: 3.2/10\n"
        "- Stability index: 0.78\n"
        "- Fatigue pattern detected: Yes\n"
        "Provide phenotype likelihood, severity, and clinical notes."
    ),
)
print(test_result)


# %% [Cell 5] — Start Flask API Server with ngrok Tunnel
# =========================================================
from flask import Flask, request, jsonify
from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient
import threading

# --- Retrieve secrets ---
secrets = UserSecretsClient()
NGROK_AUTH_TOKEN = secrets.get_secret("NGROK_AUTH_TOKEN")

# --- Flask app ---
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_ID})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
        "system_prompt": "...",
        "user_prompt": "...",
        "max_new_tokens": 1024   (optional)
    }
    Returns JSON:
    {
        "generated_text": "..."
    }
    """
    data = request.json
    system_prompt = data.get("system_prompt", "")
    user_prompt   = data.get("user_prompt", "")
    max_tokens    = data.get("max_new_tokens", 1024)

    result = generate_response(system_prompt, user_prompt, max_tokens)
    return jsonify({"generated_text": result})

# --- Open ngrok tunnel ---
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(5000)

print("=" * 60)
print(f"🔗 PUBLIC URL: {public_url}")
print("=" * 60)
print()
print("Copy the URL above and set it in your backend:")
print(f'  set KAGGLE_MEDGEMMA_URL={public_url}/predict')
print()
print("Keep this notebook running while using the API.")
print("=" * 60)

# --- Run Flask in background thread ---
threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=5000),
    daemon=True,
).start()


# %% [Cell 6] — Keep Notebook Alive
# =====================================
# Run this cell to prevent the notebook from timing out.
# The API will stay live as long as this cell is running.
import time
try:
    while True:
        time.sleep(60)
        print(".", end="", flush=True)
except KeyboardInterrupt:
    print("\n🛑 Server stopped.")
