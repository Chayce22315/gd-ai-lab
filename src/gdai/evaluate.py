"""Evaluate JSON validity and output schema on held-out prompt examples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .generate import generate_text, load_model, prepare_prompt
from .schemas import validate_output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=["deconet","gdcore"],required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True,help="held-out JSONL file")
    parser.add_argument("--limit",type=int,default=5)
    parser.add_argument("--max-new-tokens",type=int,default=1200)
    parser.add_argument("--temperature",type=float,default=0.2)
    parser.add_argument("--top-k",type=int,default=1)
    parser.add_argument("--report",type=Path,default=None)
    args=parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"))
    model,payload=load_model(args.checkpoint,device)
    rows=[]
    with args.data.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
            if len(rows)>=args.limit: break
    results=[]
    valid_json=valid_schema=0
    for idx,row in enumerate(rows):
        try:
            text=generate_text(model,row["prompt"],device,args.max_new_tokens,args.temperature,args.top_k).strip()
            value=json.loads(text)
            json_ok=True
            errors=validate_output(args.model,value)
            schema_ok=not errors
        except Exception as exc:
            text=""
            json_ok=False
            schema_ok=False
            errors=[f"{type(exc).__name__}: {exc}"]
        valid_json += int(json_ok)
        valid_schema += int(schema_ok)
        results.append({"index":idx,"task_type":row.get("task_type"),"json_valid":json_ok,"schema_valid":schema_ok,"errors":errors,"generated":text[:4000]})
    report={"model":args.model,"checkpoint":str(args.checkpoint),"from_scratch":payload.get("from_scratch"),"examples":len(results),"json_valid":valid_json,"json_valid_rate":valid_json/max(1,len(results)),"schema_valid":valid_schema,"schema_valid_rate":valid_schema/max(1,len(results)),"results":results,"warning":"Synthetic held-out data measures only this template task; it does not establish real-world GD skill or featured quality."}
    output=json.dumps(report,indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(output,encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()
