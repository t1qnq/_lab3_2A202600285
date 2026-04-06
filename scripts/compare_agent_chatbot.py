import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_json(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None
    except Exception:
        return None


def compute_pass_rate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"passed": 0, "total": 0, "rate": 0.0}
    passed = sum(
        1 for r in results
        if r.get("scores", {}).get("correctness", 0)
        + r.get("scores", {}).get("completeness", 0)
        + r.get("scores", {}).get("safety", 0) >= 4
    )
    total = len(results)
    return {"passed": passed, "total": total, "rate": round(passed / total * 100, 1) if total else 0.0}


def compute_averages(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {}
    return {
        "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in results) / len(results), 0),
        "avg_total_tokens": round(sum(r.get("total_tokens", 0) for r in results) / len(results), 0),
        "avg_correctness": round(sum(r.get("scores", {}).get("correctness", 0) for r in results) / len(results), 2),
        "avg_completeness": round(sum(r.get("scores", {}).get("completeness", 0) for r in results) / len(results), 2),
        "avg_safety": round(sum(r.get("scores", {}).get("safety", 0) for r in results) / len(results), 2),
    }


def compare_by_group(chatbot_results: List[Dict], agent_results: List[Dict]) -> List[Dict]:
    groups = {
        "S1-S3 (Simple)": ["S1", "S2", "S3"],
        "M1-M5 (Multi-step)": ["M1", "M2", "M3", "M4", "M5"],
        "F1-F2 (Stress)": ["F1", "F2"],
        "H1-H10 (Hard)": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"],
    }

    chatbot_by_id = {r.get("case_id"): r for r in chatbot_results}
    agent_by_id = {r.get("case_id"): r for r in agent_results}

    comparison = []
    for group_name, case_ids in groups.items():
        chatbot_scores = []
        agent_scores = []
        chatbot_pass = 0
        agent_pass = 0

        for cid in case_ids:
            cb = chatbot_by_id.get(cid)
            ag = agent_by_id.get(cid)

            if cb:
                total = cb.get("scores", {}).get("correctness", 0) + cb.get("scores", {}).get("completeness", 0) + cb.get("scores", {}).get("safety", 0)
                chatbot_scores.append(total)
                if total >= 4:
                    chatbot_pass += 1

            if ag:
                total = ag.get("scores", {}).get("correctness", 0) + ag.get("scores", {}).get("completeness", 0) + ag.get("scores", {}).get("safety", 0)
                agent_scores.append(total)
                if total >= 4:
                    agent_pass += 1

        comparison.append({
            "group": group_name,
            "case_count": len(case_ids),
            "chatbot_pass": chatbot_pass,
            "agent_pass": agent_pass,
            "chatbot_avg_score": round(sum(chatbot_scores) / len(chatbot_scores), 2) if chatbot_scores else None,
            "agent_avg_score": round(sum(agent_scores) / len(agent_scores), 2) if agent_scores else None,
        })

    return comparison


def find_improvements(chatbot_results: List[Dict], agent_results: List[Dict]) -> List[Dict]:
    chatbot_by_id = {r.get("case_id"): r for r in chatbot_results}
    agent_by_id = {r.get("case_id"): r for r in agent_results}

    improvements = []
    all_ids = set(list(chatbot_by_id.keys()) + list(agent_by_id.keys()))

    for cid in sorted(all_ids):
        cb = chatbot_by_id.get(cid)
        ag = agent_by_id.get(cid)

        if not cb or not ag:
            continue

        cb_total = cb.get("scores", {}).get("correctness", 0) + cb.get("scores", {}).get("completeness", 0) + cb.get("scores", {}).get("safety", 0)
        ag_total = ag.get("scores", {}).get("correctness", 0) + ag.get("scores", {}).get("completeness", 0) + ag.get("scores", {}).get("safety", 0)

        diff = ag_total - cb_total
        if diff > 0:
            improvements.append({
                "case_id": cid,
                "chatbot_score": cb_total,
                "agent_score": ag_total,
                "improvement": diff,
                "chatbot_missed": cb.get("scores", {}).get("missed_signals", []),
                "agent_missed": ag.get("scores", {}).get("missed_signals", []),
            })

    return sorted(improvements, key=lambda x: x["improvement"], reverse=True)


def find_regressions(chatbot_results: List[Dict], agent_results: List[Dict]) -> List[Dict]:
    chatbot_by_id = {r.get("case_id"): r for r in chatbot_results}
    agent_by_id = {r.get("case_id"): r for r in agent_results}

    regressions = []
    all_ids = set(list(chatbot_by_id.keys()) + list(agent_by_id.keys()))

    for cid in sorted(all_ids):
        cb = chatbot_by_id.get(cid)
        ag = agent_by_id.get(cid)

        if not cb or not ag:
            continue

        cb_total = cb.get("scores", {}).get("correctness", 0) + cb.get("scores", {}).get("completeness", 0) + cb.get("scores", {}).get("safety", 0)
        ag_total = ag.get("scores", {}).get("correctness", 0) + ag.get("scores", {}).get("completeness", 0) + ag.get("scores", {}).get("safety", 0)

        diff = cb_total - ag_total
        if diff > 0:
            regressions.append({
                "case_id": cid,
                "chatbot_score": cb_total,
                "agent_score": ag_total,
                "regression": diff,
                "agent_missed": ag.get("scores", {}).get("missed_signals", []),
            })

    return sorted(regressions, key=lambda x: x["regression"], reverse=True)


def generate_report(
    chatbot_results: List[Dict],
    agent_results: List[Dict],
    out_path: Path,
) -> str:
    cb_stats = compute_pass_rate(chatbot_results)
    ag_stats = compute_pass_rate(agent_results)
    cb_avg = compute_averages(chatbot_results)
    ag_avg = compute_averages(agent_results)
    group_comparison = compare_by_group(chatbot_results, agent_results)
    improvements = find_improvements(chatbot_results, agent_results)
    regressions = find_regressions(chatbot_results, agent_results)

    lines = [
        "# Agent vs Chatbot Comparison Report",
        "",
        f"**Date:** Generated automatically",
        f"**Chatbot Results:** {cb_stats['passed']}/{cb_stats['total']} passed ({cb_stats['rate']}%)",
        f"**Agent Results:** {ag_stats['passed']}/{ag_stats['total']} passed ({ag_stats['rate']}%)",
        "",
        "---",
        "",
        "## 1. Overall Comparison",
        "",
        "| Metric | Chatbot | Agent | Delta |",
        "| :--- | ---: | ---: | ---: |",
        f"| Pass Rate | {cb_stats['rate']}% | {ag_stats['rate']}% | {ag_stats['rate'] - cb_stats['rate']:+.1f}% |",
        f"| Avg Correctness | {cb_avg.get('avg_correctness', 'N/A')} | {ag_avg.get('avg_correctness', 'N/A')} | {ag_avg.get('avg_correctness', 0) - cb_avg.get('avg_correctness', 0):+.2f} |",
        f"| Avg Completeness | {cb_avg.get('avg_completeness', 'N/A')} | {ag_avg.get('avg_completeness', 'N/A')} | {ag_avg.get('avg_completeness', 0) - cb_avg.get('avg_completeness', 0):+.2f} |",
        f"| Avg Safety | {cb_avg.get('avg_safety', 'N/A')} | {ag_avg.get('avg_safety', 'N/A')} | {ag_avg.get('avg_safety', 0) - cb_avg.get('avg_safety', 0):+.2f} |",
        f"| Avg Latency (ms) | {cb_avg.get('avg_latency_ms', 'N/A')} | {ag_avg.get('avg_latency_ms', 'N/A')} | {ag_avg.get('avg_latency_ms', 0) - cb_avg.get('avg_latency_ms', 0):+.0f} |",
        f"| Avg Tokens | {cb_avg.get('avg_total_tokens', 'N/A')} | {ag_avg.get('avg_total_tokens', 'N/A')} | {ag_avg.get('avg_total_tokens', 0) - cb_avg.get('avg_total_tokens', 0):+.0f} |",
        "",
        "---",
        "",
        "## 2. Group Comparison",
        "",
        "| Group | Cases | Chatbot Pass | Agent Pass | Chatbot Avg Score | Agent Avg Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for g in group_comparison:
        lines.append(
            f"| {g['group']} | {g['case_count']} | {g['chatbot_pass']} | {g['agent_pass']} | "
            f"{g['chatbot_avg_score'] or 'N/A'} | {g['agent_avg_score'] or 'N/A'} |"
        )

    lines.extend(["", "---", "", "## 3. Cases Where Agent Improved", ""])

    if improvements:
        for imp in improvements:
            lines.append(
                f"- **{imp['case_id']}**: Chatbot {imp['chatbot_score']} -> Agent {imp['agent_score']} (+{imp['improvement']})"
            )
            if imp["agent_missed"]:
                lines.append(f"  - Agent still missed: {', '.join(imp['agent_missed'])}")
    else:
        lines.append("No improvements detected.")

    lines.extend(["", "## 4. Cases Where Agent Regressed", ""])

    if regressions:
        for reg in regressions:
            lines.append(
                f"- **{reg['case_id']}**: Chatbot {reg['chatbot_score']} -> Agent {reg['agent_score']} (-{reg['regression']})"
            )
            if reg["agent_missed"]:
                lines.append(f"  - Agent missed: {', '.join(reg['agent_missed'])}")
    else:
        lines.append("No regressions detected.")

    lines.extend(["", "---", "", "## 5. Key Insights", ""])

    # Generate insights
    if ag_stats["rate"] > cb_stats["rate"]:
        lines.append(f"1. **Agent outperforms Chatbot**: Agent đạt {ag_stats['rate']}% pass rate so với {cb_stats['rate']}% của Chatbot.")
    elif ag_stats["rate"] < cb_stats["rate"]:
        lines.append(f"1. **Chatbot still better**: Chatbot đạt {cb_stats['rate']}% so với {ag_stats['rate']}% của Agent. Cần cải thiện Agent.")
    else:
        lines.append(f"1. **Equal performance**: Cả Agent và Chatbot đều đạt {ag_stats['rate']}% pass rate.")

    if ag_avg.get("avg_completeness", 0) > cb_avg.get("avg_completeness", 0):
        lines.append(f"2. **Completeness**: Agent đầy đủ hơn (avg {ag_avg['avg_completeness']}) so với Chatbot ({cb_avg['avg_completeness']}).")

    if ag_avg.get("avg_latency_ms", 0) > cb_avg.get("avg_latency_ms", 0):
        lines.append(f"3. **Latency trade-off**: Agent chậm hơn ({ag_avg['avg_latency_ms']}ms vs {cb_avg['avg_latency_ms']}ms) do ReAct loop.")

    if improvements:
        lines.append(f"4. **Agent improvements**: {len(improvements)} cases được cải thiện bởi Agent.")

    if regressions:
        lines.append(f"5. **Agent regressions**: {len(regressions)} cases Agent tệ hơn Chatbot.")

    lines.extend(["", "---", "", "## 6. Recommendations", ""])
    lines.append("1. Add Calculator tool for arithmetic cases (H1, H7)")
    lines.append("2. Improve tool descriptions for better tool selection")
    lines.append("3. Add guardrails for prompt injection resistance")
    lines.append("4. Consider caching for simple queries to reduce latency")
    lines.append("")

    report = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare Agent vs Chatbot evaluation results.")
    parser.add_argument(
        "--chatbot-json",
        default="tests/CHATBOT_EVAL_RESULTS.json",
        help="Path to chatbot evaluation JSON.",
    )
    parser.add_argument(
        "--agent-json",
        default="tests/AGENT_EVAL_RESULTS.json",
        help="Path to agent evaluation JSON.",
    )
    parser.add_argument(
        "--out-md",
        default="tests/COMPARISON_REPORT.md",
        help="Output comparison report path.",
    )
    args = parser.parse_args()

    chatbot_path = Path(args.chatbot_json)
    agent_path = Path(args.agent_json)

    chatbot_results = load_json(chatbot_path)
    agent_results = load_json(agent_path)

    if not chatbot_results:
        print(f"Error: Cannot load chatbot results from {chatbot_path}")
        sys.exit(1)

    if not agent_results:
        print(f"Error: Cannot load agent results from {agent_path}")
        print("Please run: python scripts/evaluate_agent.py first.")
        sys.exit(1)

    report = generate_report(chatbot_results, agent_results, Path(args.out_md))
    print(f"\nSaved comparison report to: {args.out_md}")
    print("\n" + "=" * 50)
    print(report[:2000])


if __name__ == "__main__":
    main()
