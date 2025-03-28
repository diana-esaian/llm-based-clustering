# LLM-Based Question Clustering
> A Python-based solution that assigns a general topic (cluster) to each customer message without using embeddings. Instead, it relies solely on prompt chaining.

This implementation utilizes Google's Gemini model.

## Repository Structure
- `prompt_rewriting.ipynb` – a notebook that refines input prompts using an LLM to align the input distribution with the model’s training data, reducing entropy and bias.

- `prompts.py` – a script containing all the predefined prompts used in the process.

- `pipeline.py` – a structured, object-oriented implementation of the clustering pipeline, based on the notebook.

- `output.json` – a JSON file summarizing the detected message clusters.