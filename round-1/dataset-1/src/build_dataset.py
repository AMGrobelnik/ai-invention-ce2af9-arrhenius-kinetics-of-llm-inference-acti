#!/usr/bin/env python3
"""Build MMLU-Pro splits for Arrhenius inference-energy experiment."""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/build.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path("/ai-inventor/aii_data/runs/run_wYelBzy-9k_d/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
SEED = 42
PILOT_N = 200
MAIN_N = 500
CATALYSIS_N = 50

VALID_LETTERS = set("ABCDEFGHIJ")


@logger.catch(reraise=True)
def main():
    from datasets import load_dataset

    logger.info("Loading TIGER-Lab/MMLU-Pro test split...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    raw_count = len(ds)
    logger.info(f"Raw rows: {raw_count}")

    # Convert to list of dicts
    rows = [dict(r) for r in ds]

    # ── Filters ────────────────────────────────────────────────────────────────
    def f1(r):
        opts = r.get("options", [])
        return len(opts) >= 2 and all(o.strip() for o in opts)

    def f2(r):
        ans = r.get("answer", "")
        idx = r.get("answer_index", -1)
        return len(ans) == 1 and ans.isupper() and idx == ord(ans) - ord("A")

    def f3(r):
        return r.get("answer", "") in VALID_LETTERS

    def f4(r):
        opts = r.get("options", [])
        idx = r.get("answer_index", -1)
        return 0 <= idx < len(opts) and opts[idx].strip()

    after = {"raw": raw_count}
    rows = [r for r in rows if f1(r)]; after["after_F1"] = len(rows); logger.info(f"After F1: {len(rows)}")
    rows = [r for r in rows if f2(r)]; after["after_F2"] = len(rows); logger.info(f"After F2: {len(rows)}")
    rows = [r for r in rows if f3(r)]; after["after_F3"] = len(rows); logger.info(f"After F3: {len(rows)}")
    rows = [r for r in rows if f4(r)]; after["after_F4"] = len(rows); logger.info(f"After F4: {len(rows)}")

    total_filtered = len(rows)
    subjects = sorted(set(r["category"] for r in rows))
    logger.info(f"Subjects ({len(subjects)}): {subjects}")

    # ── Splits ─────────────────────────────────────────────────────────────────
    random.seed(SEED)
    all_ids = list(range(total_filtered))

    # pilot_set
    pilot_ids = set(random.sample(all_ids, PILOT_N))
    remaining_ids = [i for i in all_ids if i not in pilot_ids]
    logger.info(f"Pilot set: {PILOT_N}, remaining: {len(remaining_ids)}")

    # main_set – stratified by subject
    cats = defaultdict(list)
    for i in remaining_ids:
        cats[rows[i]["category"]].append(i)

    per_cat = {c: max(1, round(MAIN_N * len(v) / len(remaining_ids))) for c, v in cats.items()}
    # Adjust sum to exactly MAIN_N
    delta = MAIN_N - sum(per_cat.values())
    if delta != 0:
        keys = sorted(per_cat, key=lambda c: per_cat[c], reverse=(delta > 0))
        for k in keys:
            if delta == 0:
                break
            per_cat[k] += 1 if delta > 0 else -1
            delta += -1 if delta > 0 else 1

    main_ids = []
    for c, idxs in cats.items():
        n = min(per_cat[c], len(idxs))
        main_ids.extend(random.sample(idxs, n))

    # top-up or trim
    if len(main_ids) < MAIN_N:
        leftover = [i for i in remaining_ids if i not in set(main_ids)]
        needed = MAIN_N - len(main_ids)
        main_ids.extend(random.sample(leftover, min(needed, len(leftover))))
    main_ids = main_ids[:MAIN_N]
    logger.info(f"Main set size: {len(main_ids)}")

    # catalysis_set – subset of main
    catalysis_local_ids = random.sample(range(len(main_ids)), CATALYSIS_N)
    catalysis_ids = [main_ids[i] for i in catalysis_local_ids]
    catalysis_set_global = set(catalysis_ids)
    logger.info(f"Catalysis set size: {len(catalysis_ids)}")

    # ── Row builder ────────────────────────────────────────────────────────────
    def build_row(r, split_label):
        opts = r["options"]
        ans_idx = r["answer_index"]
        ans_letter = r["answer"]
        wrong_idx = 0 if ans_idx != 0 else 1
        wrong_letter = chr(ord("A") + wrong_idx)

        return {
            "question_id": int(r["question_id"]),
            "subject": r["category"],
            "question_text": r["question"],
            "choices": opts,
            "num_choices": len(opts),
            "correct_answer_letter": ans_letter,
            "correct_answer_index": int(ans_idx),
            "correct_answer_text": opts[ans_idx],
            "wrong_answer_letter": wrong_letter,
            "wrong_answer_index": wrong_idx,
            "wrong_answer_text": opts[wrong_idx],
            "answer_token_id_hint": None,
            "token_hint_note": "compute at experiment runtime using AutoTokenizer",
            "cot_content": r.get("cot_content", ""),
            "src": r.get("src", ""),
            "split": split_label,
        }

    pilot_set = [build_row(rows[i], "pilot_set") for i in sorted(pilot_ids)]
    main_set = [
        build_row(rows[i], "catalysis_set" if i in catalysis_set_global else "main_set")
        for i in main_ids
    ]
    catalysis_set = [build_row(rows[i], "catalysis_set") for i in catalysis_ids]

    logger.info(f"Built pilot={len(pilot_set)}, main={len(main_set)}, catalysis={len(catalysis_set)}")

    # ── Write filter log ────────────────────────────────────────────────────────
    filter_log_path = WORKSPACE / "filter_log.txt"
    with filter_log_path.open("w") as f:
        f.write("MMLU-Pro Filter Log\n")
        f.write("=" * 40 + "\n")
        for k, v in after.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nSubjects ({len(subjects)}):\n")
        for s in subjects:
            f.write(f"  {s}\n")
        f.write(f"\nSplit sizes:\n  pilot_set: {len(pilot_set)}\n  main_set: {len(main_set)}\n  catalysis_set: {len(catalysis_set)}\n")
    logger.info(f"Wrote {filter_log_path}")

    # ── Write schema.json ───────────────────────────────────────────────────────
    schema = {
        "type": "object",
        "required": ["question_id", "subject", "question_text", "choices", "num_choices",
                     "correct_answer_letter", "correct_answer_index", "correct_answer_text", "split"],
        "properties": {
            "question_id": {"type": "integer"},
            "subject": {"type": "string"},
            "question_text": {"type": "string", "minLength": 5},
            "choices": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "num_choices": {"type": "integer", "minimum": 2},
            "correct_answer_letter": {"type": "string", "pattern": "^[A-J]$"},
            "correct_answer_index": {"type": "integer", "minimum": 0, "maximum": 9},
            "correct_answer_text": {"type": "string", "minLength": 1},
            "split": {"type": "string", "enum": ["pilot_set", "main_set", "catalysis_set"]},
        }
    }
    schema_path = WORKSPACE / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    logger.info(f"Wrote schema.json")

    # ── Validate all rows ───────────────────────────────────────────────────────
    import jsonschema
    errors = []
    for split_name, split_rows in [("pilot_set", pilot_set), ("main_set", main_set), ("catalysis_set", catalysis_set)]:
        for i, row in enumerate(split_rows):
            try:
                jsonschema.validate(row, schema)
            except jsonschema.ValidationError as e:
                errors.append(f"{split_name}[{i}]: {e.message}")
    if errors:
        logger.error(f"Schema validation errors: {errors[:5]}")
        raise ValueError(f"{len(errors)} schema violations")
    logger.info("All rows pass schema validation")

    # ── Assemble data_out.json ──────────────────────────────────────────────────
    output = {
        "metadata": {
            "source": "TIGER-Lab/MMLU-Pro",
            "hf_splits_used": ["test"],
            "total_filtered": total_filtered,
            "seed": SEED,
            "schema_version": "1.0",
            "subjects": subjects,
            "filter_counts": after,
        },
        "pilot_set": pilot_set,
        "main_set": main_set,
        "catalysis_set": catalysis_set,
    }

    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Wrote data_out.json ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ── Preview and mini variants ───────────────────────────────────────────────
    def truncate_strings(obj, max_len=200):
        if isinstance(obj, str):
            return obj[:max_len] + "..." if len(obj) > max_len else obj
        elif isinstance(obj, list):
            return [truncate_strings(x, max_len) for x in obj]
        elif isinstance(obj, dict):
            return {k: truncate_strings(v, max_len) for k, v in obj.items()}
        return obj

    preview_output = {
        "metadata": output["metadata"],
        "pilot_set": truncate_strings(pilot_set[:3]),
        "main_set": truncate_strings(main_set[:3]),
        "catalysis_set": truncate_strings(catalysis_set[:3]),
    }
    preview_path = WORKSPACE / "data_out_preview.json"
    preview_path.write_text(json.dumps(preview_output, indent=2, ensure_ascii=False))
    logger.info(f"Wrote data_out_preview.json")

    mini_output = {
        "metadata": output["metadata"],
        "pilot_set": pilot_set[:10],
        "main_set": main_set[:10],
        "catalysis_set": catalysis_set[:10],
    }
    mini_path = WORKSPACE / "data_out_mini.json"
    mini_path.write_text(json.dumps(mini_output, indent=2, ensure_ascii=False))
    logger.info(f"Wrote data_out_mini.json")

    logger.info("Done.")
    return output


if __name__ == "__main__":
    main()
