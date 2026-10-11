# Deploy this folder as a Hugging Face Space (2 minutes)

1. huggingface.co → New → Space → name `BladeDelta`, SDK **Gradio**, hardware **CPU basic (free)**.
2. Upload every file in this folder (or `huggingface-cli upload <user>/BladeDelta . --repo-type space` with a **write** token).
The Space shows the five precomputed examples; live analysis needs a GPU (run `demo/app.py` locally or use the notebook).
