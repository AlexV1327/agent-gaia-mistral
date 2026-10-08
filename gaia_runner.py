import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from agent.runner import GaiaAgent


def normalize(text: str) -> str:
    return (
        str(text)
        .lower()
        .strip()
        .strip('"')
        .strip("'")
        .strip("«")
        .strip("»")
        .strip()
    )


def is_correct(prediction: str, expected: str | None) -> bool | None:
    if expected is None:
        return None

    return normalize(prediction) == normalize(expected)


def error_kind(error: str | None) -> str | None:
    if not error:
        return None

    lowered = error.lower()
    if any(marker in lowered for marker in ("429", "rate limit", "rate_limited", "capacity")):
        return "rate_limit"
    if any(marker in lowered for marker in ("timeout", "timed out")):
        return "timeout"
    if any(marker in lowered for marker in ("502", "503", "504", "service unavailable")):
        return "temporary_api_error"
    return "runtime_error"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local agent on GAIA validation questions.")

    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to run. Default: validation.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the selected dataset. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of questions to run. Default: 2.",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Optional GAIA level filter: 1, 2, or 3.",
    )
    parser.add_argument(
        "--planner",
        action="store_true",
        help="Enable the planner before answering.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample random questions after split/level filtering instead of using --start order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible --random runs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path. Default: runs/gaia_<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Mistral model override. Default: MISTRAL_MODEL env var or mistral-small-latest.",
    )
    return parser.parse_args()


def load_gaia(split: str):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN introuvable dans .env")

    return load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        split=split,
        token=token,
    )


def item_level(item: dict) -> int | None:
    value = item.get("Level") or item.get("level")
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def item_id(item: dict, fallback_index: int) -> str:
    return str(
        item.get("task_id")
        or item.get("Task ID")
        or item.get("id")
        or fallback_index
    )


def resolve_attachment(item: dict) -> str | None:
    file_path = item.get("file_path")
    if not file_path:
        return None

    token = os.getenv("HF_TOKEN")
    try:
        return hf_hub_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            filename=file_path,
            token=token,
        )
    except Exception as exc:
        print(f"Attention: fichier attaché non récupéré ({file_path}): {exc}")
        return None


def question_with_attachment(question: str, attachment_path: str | None) -> str:
    if not attachment_path:
        return question

    return (
        f"{question}\n\n"
        f"Fichier attaché local: {attachment_path}\n"
        "Lis ce fichier avec l'outil approprié avant de répondre."
    )


def main():
    load_dotenv()
    args = parse_args()

    dataset = load_gaia(args.split)

    items = list(dataset)

    if args.level is not None:
        items = [item for item in items if item_level(item) == args.level]

    indexed_items = list(enumerate(items))

    if args.random:
        rng = random.Random(args.seed)
        sample_size = min(args.limit, len(indexed_items))
        selected = rng.sample(indexed_items, sample_size)
    else:
        selected = indexed_items[args.start : args.start + args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or f"runs/gaia_{timestamp}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agent = GaiaAgent(model=args.model)

    print(f"Split: {args.split}")
    print(f"Questions disponibles après filtre: {len(items)}")
    print(f"Début: {args.start}")
    print(f"Limite: {args.limit}")
    print(f"Random: {args.random}")
    if args.random:
        print(f"Seed: {args.seed}")
    print(f"Planner: {args.planner}")
    print(f"Model: {agent.llm.model}")
    print(f"Sortie: {output_path}")

    score = 0
    total = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for dataset_index, item in selected:
            question = item["Question"]
            expected = item.get("Final answer")
            level = item_level(item)
            task_id = item_id(item, dataset_index)
            attachment_path = resolve_attachment(item)
            agent_question = question_with_attachment(question, attachment_path)

            print("\n====================")
            print(f"Index: {dataset_index}")
            print(f"Task ID: {task_id}")
            print(f"Level: {level}")
            print(f"Question: {question}")
            if attachment_path:
                print(f"Fichier attaché: {attachment_path}")
            print(f"Réponse attendue: {expected}")

            error = None
            prediction = ""

            try:
                prediction = agent.run(agent_question, use_planner=args.planner)
                print(f"Réponse agent: {prediction}")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"Erreur: {error}")

            correct = is_correct(prediction, expected)

            if correct is not None:
                total += 1
                if correct:
                    score += 1
                    print("Correct: oui")
                else:
                    print("Correct: non")

            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "split": args.split,
                "index": dataset_index,
                "task_id": task_id,
                "level": level,
                "question": question,
                "attachment_path": attachment_path,
                "expected": expected,
                "prediction": prediction,
                "correct": correct,
                "error": error,
                "error_kind": error_kind(error),
                "model": agent.llm.model,
                "trace": agent.last_trace,
            }


            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    print("\n====================")
    print(f"Score: {score}/{total}")
    print(f"Résultats sauvegardés dans: {output_path}")


if __name__ == "__main__":
    main()
