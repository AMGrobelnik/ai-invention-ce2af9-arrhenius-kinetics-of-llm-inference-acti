#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru"]
# ///
"""Convert MMLU-Pro data_out.json to exp_sel_data_out.json schema."""

import json
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/data.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path("/ai-inventor/aii_data/runs/run_wYelBzy-9k_d/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
LETTERS = "ABCDEFGHIJ"


def format_mcqa_input(question_text: str, choices: list[str]) -> str:
    """Format question + choices as a MCQA prompt string."""
    lines = [question_text.strip(), ""]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    return "\n".join(lines)


def row_to_example(row: dict) -> dict:
    """Convert a data_out.json row to an exp_sel_data_out example."""
    input_str = format_mcqa_input(row["question_text"], row["choices"])
    output_str = row["correct_answer_letter"]
    return {
        "input": input_str,
        "output": output_str,
        "metadata_question_id": row["question_id"],
        "metadata_subject": row["subject"],
        "metadata_num_choices": row["num_choices"],
        "metadata_correct_answer_index": row["correct_answer_index"],
        "metadata_correct_answer_text": row["correct_answer_text"],
        "metadata_wrong_answer_letter": row["wrong_answer_letter"],
        "metadata_wrong_answer_index": row["wrong_answer_index"],
        "metadata_wrong_answer_text": row["wrong_answer_text"],
        "metadata_split": row["split"],
        "metadata_src": row.get("src", ""),
        "metadata_task_type": "multiple_choice_qa",
    }


def main() -> None:
    data_path = WORKSPACE / "data_out.json"
    logger.info(f"Loading {data_path}")
    data = json.loads(data_path.read_text())
    meta = data["metadata"]

    # Collect all unique rows: pilot_set + main_set (catalysis rows already in main)
    # Use question_id to deduplicate
    seen_ids: set[int] = set()
    all_rows: list[dict] = []
    for split_key in ("pilot_set", "main_set", "catalysis_set"):
        for row in data[split_key]:
            qid = row["question_id"]
            if qid not in seen_ids:
                seen_ids.add(qid)
                all_rows.append(row)

    logger.info(f"Total unique rows: {len(all_rows)}")

    examples = [row_to_example(r) for r in all_rows]
    logger.info(f"Converted {len(examples)} examples")

    output = {
        "metadata": {
            "source": meta["source"],
            "hf_splits_used": meta["hf_splits_used"],
            "total_filtered": meta["total_filtered"],
            "seed": meta["seed"],
            "schema_version": meta["schema_version"],
            "subjects": meta["subjects"],
            "filter_counts": meta["filter_counts"],
            "description": (
                "MMLU-Pro MCQA benchmark: 12032-item 14-subject dataset for Arrhenius "
                "inference-energy experiment. Each example is a formatted multiple-choice "
                "question with up to 10 options; output is the correct answer letter (A-J)."
            ),
        },
        "datasets": [
            {
                "dataset": "TIGER-Lab/MMLU-Pro",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Wrote {out_path} ({size_mb:.1f} MB, {len(examples)} examples)")


if __name__ == "__main__":
    main()
