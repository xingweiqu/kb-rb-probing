"""Extract hidden states from model."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_hidden_states(
    model_name: str,
    dataset_path: str,
    output_path: str,
    device: str = "cuda"
) -> dict[str, dict[str, Any]]:
    """
    Extract hidden states for all items.

    Returns:
        states: dict[family_id][variant] -> [n_layers, hidden_dim]
    """
    logger.info(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    items = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    logger.info(f"Loaded {len(items)} items")

    # Extract states
    states = {}
    model_outputs = []

    with torch.no_grad():
        for i, item in enumerate(items):
            if i % 10 == 0:
                logger.info(f"Processing {i}/{len(items)}...")

            question = item["question"]
            inputs = tokenizer(question, return_tensors="pt").to(device)

            outputs = model(**inputs, output_hidden_states=True)

            # Extract last token hidden states for all layers
            hidden = torch.stack([h[:, -1, :] for h in outputs.hidden_states])  # [n_layers, 1, hidden_dim]
            hidden = hidden.squeeze(1).cpu().numpy()  # [n_layers, hidden_dim]

            # Get prediction
            logits = outputs.logits[:, -1, :]
            pred_token_id = logits.argmax(dim=-1).item()
            pred_text = tokenizer.decode([pred_token_id])

            # Store
            family_id = item["family_id"]
            variant = item["variant"]
            mode = item.get("mode", "natural")

            key = f"{family_id}_{variant}_{mode}"
            states[key] = hidden

            model_outputs.append({
                **item,
                "prediction": pred_text,
                "correct": pred_text.strip().lower() == item["gold_answer"].strip().lower()
            })

    # Save
    logger.info(f"Saving hidden states to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(states, f)

    # Save model outputs
    output_file = Path(output_path).parent / "model_outputs.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in model_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(states)} hidden states and {len(model_outputs)} predictions")
    return states


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset JSONL file")
    parser.add_argument("--output", required=True, help="Output pickle file")
    parser.add_argument("--device", default="cuda", help="Device")

    args = parser.parse_args()

    extract_hidden_states(
        model_name=args.model_name,
        dataset_path=args.dataset,
        output_path=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()
