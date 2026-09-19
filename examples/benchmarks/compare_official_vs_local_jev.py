#!/usr/bin/env python3
"""Compare Official TypeSafe Jev (Cloud System One API) vs. Local DeepSeek-V4.1 Jev Mode.

Evaluates:
1. Decision accuracy and semantic alignment (Urgency, Category, Sentiment/Score, Action).
2. Probability calibration & confidence reporting.
3. Latency (Cloud round-trip vs. Local 4x A100 GPU cluster).
4. Architectural differences and capabilities.
"""

import json
import os
import sys
import time
import urllib.request

TYPESAFE_API_KEY = os.environ.get(
    "TYPESAFE_API_KEY",
    "apikey_213088f1322b5885496b999b159a452ee161_0920d4a50e53f268ee01015aac9f679af6d74bb8d467312bc1bbaeb51ad85cad",
)
LOCAL_SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")

TEST_CASES = [
    {
        "name": "Urgent Bug & Churn Threat",
        "text": (
            "I have been a paying Pro user for 2 years. After the latest update, "
            "the export function completely crashes my browser. We have a major client "
            "presentation tomorrow morning and cannot export our work! Fix this immediately "
            "or we are canceling our enterprise migration."
        ),
    },
    {
        "name": "Routine Billing Query",
        "text": (
            "Hi team, where can I download our VAT invoice for the annual subscription "
            "renewed on September 15th? Our finance department needs it for tax filing. Thanks!"
        ),
    },
    {
        "name": "Feature Request & Ergonomics",
        "text": (
            "I love the clean UI! Would it be possible to add a true dark mode and "
            "customizable keyboard shortcuts for power users? It would make long coding sessions "
            "much more comfortable."
        ),
    },
    {
        "name": "Sarcastic / Mixed Feedback",
        "text": (
            "Oh wonderful, another 'revolutionary' redesign that hid all the buttons "
            "I use every day behind three sub-menus. Brilliant job, guys. At least the server "
            "didn't crash this time."
        ),
    },
]


def call_official_jev(text: str) -> dict:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    client = TypeSafeClient(api_key=TYPESAFE_API_KEY)
    t0 = time.perf_counter()
    resp = client.system_one(
        state=text,
        questions={
            "is_urgent": Noul(
                instructions="Does this customer message convey genuine urgency or critical deadline?",
                criteria={"true": "Time-sensitive or blocking issue", "false": "Routine or non-urgent inquiry"},
            ),
            "category": Choice(
                instructions="What is the primary category of this inquiry?",
                criteria={
                    "bug": "Software defect, error, or crash",
                    "billing": "Invoice, payment, or subscription question",
                    "feature_request": "Suggestion or request for new feature",
                    "usability": "Complaint or feedback regarding UI / UX",
                },
            ),
            "frustration": Score(
                instructions="Rate customer frustration from calm to furious",
                criteria=["Calm / Positive", "Mildly annoyed / Constructive", "Angry / Threatening cancellation"],
            ),
            "suggested_action": Choice(
                instructions="What is the recommended immediate operational action?",
                criteria={
                    "tech_escalation": "Escalate immediately to engineering",
                    "support_reply": "Send standard knowledge-base reply",
                    "retention_outreach": "Account manager reach out to prevent churn",
                    "product_backlog": "Log into product enhancement backlog",
                },
            ),
        },
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ans = resp.answers

    # Extract clean summary
    return {
        "elapsed_ms": elapsed_ms,
        "is_urgent_prob": getattr(ans.get("is_urgent"), "noul", None),
        "category": getattr(ans.get("category"), "choice", None),
        "category_conf": getattr(ans.get("category"), "confidence", None),
        "category_probs": getattr(ans.get("category"), "probabilities", {}),
        "frustration_score": getattr(ans.get("frustration"), "score", None),
        "frustration_conf": getattr(ans.get("frustration"), "confidence", None),
        "suggested_action": getattr(ans.get("suggested_action"), "choice", None),
        "suggested_action_conf": getattr(ans.get("suggested_action"), "confidence", None),
        "usage": getattr(resp, "usage", None),
    }


def call_local_jev(text: str) -> dict:
    url = f"{LOCAL_SERVER_URL}/v1/chat/completions"
    schema = {
        "is_urgent": [True, False],
        "category": ["bug", "billing", "feature_request", "usability"],
        "frustration": ["low", "medium", "high"],
        "suggested_action": ["tech_escalation", "support_reply", "retention_outreach", "product_backlog"],
    }
    payload = {
        "model": "deepseek-v4.1-flash",
        "prompt": text,
        "jev": True,
        "schema": schema,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ext = res.get("jev_result", {})
    return {
        "elapsed_ms": elapsed_ms,
        "is_urgent": ext.get("is_urgent"),
        "category": ext.get("category"),
        "frustration": ext.get("frustration"),
        "suggested_action": ext.get("suggested_action"),
        "prefill_tokens": res.get("prefill_tokens", 0),
        "output_tokens": res.get("tokens", 0),
    }


def main():
    print("=" * 80)
    print("Head-to-Head Benchmark: Official TypeSafe Jev vs. Local DeepSeek-V4.1 Jev Mode")
    print(f"Official API Endpoint : https://api.typesafe.ai/v1/systemone (model: jev-latest)")
    print(f"Local Server Endpoint : {LOCAL_SERVER_URL} (DeepSeek-V4.1-Flash on 4x A100)")
    print("=" * 80)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n--- [Case {i}/{len(TEST_CASES)}] {tc['name']} ---")
        print(f"Input: \"{tc['text'][:90]}...\"")

        print("  Calling Official TypeSafe Jev API...", end="", flush=True)
        try:
            off_res = call_official_jev(tc["text"])
            print(f" Done ({off_res['elapsed_ms']:.1f} ms)")
        except Exception as e:
            print(f" Failed ({e})")
            off_res = None

        print("  Calling Local DeepSeek-V4.1 Jev...", end="", flush=True)
        try:
            loc_res = call_local_jev(tc["text"])
            print(f" Done ({loc_res['elapsed_ms']:.1f} ms)")
        except Exception as e:
            print(f" Failed ({e})")
            loc_res = None

        print("\n  [Comparison Results]")
        if off_res:
            print(f"    • Official TypeSafe Jev:")
            print(f"        - Category       : {off_res['category']} (conf: {off_res['category_conf']:.2f}, probs: {off_res['category_probs']})")
            print(f"        - Urgency (Noul) : P(urgent) = {off_res['is_urgent_prob']:.2f}")
            print(f"        - Frustration    : Score = {off_res['frustration_score']:.2f}/2.0 (conf: {off_res['frustration_conf']:.2f})")
            print(f"        - Action         : {off_res['suggested_action']} (conf: {off_res['suggested_action_conf']:.2f})")
            print(f"        - Latency        : {off_res['elapsed_ms']:.1f} ms")
        if loc_res:
            print(f"    • Local DeepSeek-V4.1 Jev:")
            print(f"        - Category       : {loc_res['category']}")
            print(f"        - Urgency        : {loc_res['is_urgent']}")
            print(f"        - Frustration    : {loc_res['frustration']}")
            print(f"        - Action         : {loc_res['suggested_action']}")
            print(f"        - Latency        : {loc_res['elapsed_ms']:.1f} ms")


if __name__ == "__main__":
    main()
