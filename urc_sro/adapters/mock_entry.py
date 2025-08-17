from __future__ import annotations
import json
from urc_sro import make_mock_pipeline, make_sample_docs

def _load_cfg() -> dict:
    try:
        with open("examples/demo_config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"print_pre_post_rerank": True, "print_step_verifier_label": True, "print_citations": True}

def _inline_citations(answer: str, used_ids: list[str]) -> str:
    if not used_ids:
        return answer
    return f"{answer}  [sources: {', '.join(used_ids)}]"

def main():
    cfg = _load_cfg()
    pipeline = make_mock_pipeline(docs=make_sample_docs())

    query = "What do URC and SRO each do in this pipeline?"

    # Use the debug retriever path explicitly to get pre/post ids for printing
    docs, dbg = pipeline.urc.retrieve_with_debug(query)

    if cfg.get("print_pre_post_rerank", False):
        print("— URC DEBUG —")
        print("complexity:", round(dbg["complexity"], 3), "| k:", dbg["k"], "| sources:", dbg["sources"])
        print("pre-rerank top-k:", dbg["pre_rerank_ids"])
        print("post-rerank top-k:", dbg["post_rerank_ids"])
        print()

    # Hand the docs to SRO manually (instead of pipeline.run) so we can inspect its internals
    answer = pipeline.sro.generate_response(query, docs)

    if cfg.get("print_step_verifier_label", False):
        sup = pipeline.sro.last_step_support
        label = "entailed" if (sup and sup.entailed) else "not-entailed"
        print("— SRO DEBUG —")
        print("last-step label:", label)
        print()

    if cfg.get("print_citations", False):
        used_ids = dbg["post_rerank_ids"][:2]
        answer = _inline_citations(answer, used_ids)

    print("\n=== Final Answer ===\n")
    print(answer)

if __name__ == "__main__":
    main()
