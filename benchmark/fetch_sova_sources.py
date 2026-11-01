from pathlib import Path
import json
from datasets import load_dataset

ROOT = Path("benchmark/sova_official")
ROOT.mkdir(parents=True, exist_ok=True)

def save_jsonl(name, rows):
    p = ROOT / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(p, len(rows))

def prepare_triviaqa(target=6840):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    rows = [{"question": x.get("question"), "answer": x.get("answer"), "source": "TriviaQA"} for x in ds]
    save_jsonl("triviaqa_source", rows[:target])

def prepare_librisqa():
    ds = load_dataset("ZihanZhao/LibriSQA", split="partII.test")
    save_jsonl("librisqa_partII_test", [dict(x) for x in ds])

def prepare_spoken_squad(target=3243):
    ds = load_dataset("alinet/spoken_squad", "WER44", split="test")
    save_jsonl("spoken_squad_source", [dict(x) for x in ds][:target])

def prepare_alpaca(target=3745):
    ds = load_dataset("pss8093/alpaca", split="train")
    rows = [{"instruction":x.get("instruction"),"input":x.get("input"),
             "reference_output":x.get("output"),"source":"Alpaca"} for x in ds]
    save_jsonl("alpaca_source", rows[:target])

if __name__ == "__main__":
    print("Functions available:")
    print("prepare_triviaqa()")
    print("prepare_librisqa()")
    print("prepare_spoken_squad()")
    print("prepare_alpaca()")
    print("IEMOCAP and Common Voice need separate source-specific preparation.")
