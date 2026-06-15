"""Milestone 5 (part 2) — Gradio query interface for the Unofficial Duke Dorm Guide.

A minimal web UI: the user types a dorm question, sees the grounded answer, and sees
the source reviews it was drawn from. All the grounding/attribution logic lives in
generate.ask(); this file is just the front end.

Run:  python src/app.py   (then open the printed local URL)
"""

from __future__ import annotations

import gradio as gr

from generate import ask


def handle_query(question: str) -> tuple[str, str]:
    """Adapt generate.ask() to the two Gradio output textboxes (answer, sources)."""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"]) or "—"
    return result["answer"], sources


with gr.Blocks(title="Unofficial Duke Dorm Guide") as demo:
    gr.Markdown(
        "# 🏠 Unofficial Duke Dorm Guide\n"
        "Ask about Duke dorms — answers come **only** from real student reviews, "
        "with the source reviews listed. If the reviews don't cover it, it'll say so."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. How are the bathrooms in Wilson?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

    gr.Examples(
        examples=[
            "What are the room sizes like in Randolph?",
            "Does Basset have a mold problem?",
            "Is Blackwell or Trinity quieter?",
        ],
        inputs=inp,
    )


if __name__ == "__main__":
    demo.launch()
