# RECUR: Resource Exhaustion Attack via Recursive-Entropy Guided Counterfactual Utilization and Reflection

This is the code repository for the paper. To run the experiments, please follow the instructions below step by step.

----

- Dependency Installation

    Create the virtual environment:

    ```bash
    conda create -n Recur python=3.12.9
    ```

    Install requirements:

    ```bash
    conda activate Recur
    pip install -r requirements.txt
    ```

- Downloading Model Weights with 🤗 Transformers

    ```python
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        local_dir="./models/DeepSeek-R1-Distill-Llama-8B",
        local_dir_use_symlinks=False,
        token="YOUR_HF_TOKEN"
    )
    ```

- Environment variables and configurations

    - Set up `model` in the `config.json` for open-source models
    - `config.json` provides the default settings for the parameters
    - Set up your API key in the `config.json` for closed-source models

- Run experiments

    - Generating thinking loops

        Run `main.py` to generate counterfactual questions and over-reflection, then generate thinking loops through recursive entropy guided sampling.

        ```bash
        python src/main.py
        ```

    - Generating prompts

        Run `trim.py` to generate attack prompts.

        ```bash
        python src/trim.py
        ```

    - Reasoning length experiments

        Run `prompt.py` to generate long reasoning content.

        ```bash
        python src/prompt.py
        ```

        Run `transfer.py` to perform reasoning length experiments on the closed-source models.

        ```bash
        python src/transfer.py
        ```
