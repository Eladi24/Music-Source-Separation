"""
Stem Splitter GUI.

A small Gradio app: drop in a track, pick a separation mode, get stems back
as playable/downloadable audio. Separation itself is done by pretrained
Demucs models (see backend.py) -- nothing here is trained from scratch.
"""
import queue
import threading

import gradio as gr

from backend import DEFAULT_PRESET, MODEL_PRESETS, STEM_ORDER, separate

# state -> (background color, text color, icon)
_STATUS_STYLE = {
    "idle": ("#e5e7eb", "#374151", ""),
    "running": ("#facc15", "#1f2937", "⏳"),
    "done": ("#22c55e", "#ffffff", "✅"),
    "error": ("#ef4444", "#ffffff", "❌"),
}


def _status_html(state: str, desc: str, frac: float | None = None) -> str:
    bg, fg, icon = _STATUS_STYLE[state]
    # Only show a percentage during the "Separating" phase -- the bookend
    # steps (loading the model, saving stems) report their own fixed
    # fractions that would otherwise make the number jump backwards.
    pct = f" — {round(frac * 100)}%" if frac is not None and desc == "Separating" else ""
    return (
        '<div style="display:inline-block;padding:6px 14px;border-radius:999px;'
        f'background:{bg};color:{fg};font-weight:600;font-family:sans-serif;">'
        f"{icon} {desc}{pct}</div>"
    )


def run_separation(audio_path, preset, use_gpu):
    if audio_path is None:
        raise gr.Error("Upload or record a track first.")

    skip_stems = (gr.skip(),) * len(STEM_ORDER)

    yield (_status_html("running", "Starting"),) + skip_stems

    events: queue.Queue = queue.Queue()
    outcome = {}

    def on_progress(frac, desc):
        events.put(("progress", frac, desc))

    def worker():
        try:
            outcome["result"] = separate(
                audio_path, preset=preset, use_gpu=use_gpu, progress_callback=on_progress
            )
        except Exception as e:  # surfaced to the UI below
            outcome["error"] = e
        finally:
            events.put(("finished", None, None))

    threading.Thread(target=worker, daemon=True).start()

    last_shown = -1
    while True:
        kind, frac, desc = events.get()
        if kind == "finished":
            break
        # Only push an update when the rounded percentage actually changes,
        # so we don't flood the UI with near-identical redraws.
        pct = round(frac * 100)
        if pct != last_shown:
            last_shown = pct
            yield (_status_html("running", desc, frac),) + skip_stems

    if "error" in outcome:
        yield (_status_html("error", "Separation failed"),) + skip_stems
        raise gr.Error(str(outcome["error"]))

    result = outcome["result"]
    stem_updates = []
    for name in STEM_ORDER:
        if name in result.stems:
            stem_updates.append(gr.update(value=str(result.stems[name]), visible=True, label=name.capitalize()))
        else:
            stem_updates.append(gr.update(value=None, visible=False))

    yield (_status_html("done", "Done"),) + tuple(stem_updates)


with gr.Blocks(title="Stem Splitter") as demo:
    gr.Markdown(
        "# 🎛️ Stem Splitter\n"
        "Upload a song and split it into stems using "
        "[Demucs](https://github.com/facebookresearch/demucs), Meta's pretrained "
        "source-separation model. No training involved -- this runs a ready-made model."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(type="filepath", label="Input track")
            preset = gr.Radio(
                choices=list(MODEL_PRESETS.keys()),
                value=DEFAULT_PRESET,
                label="Separation mode",
            )
            use_gpu = gr.Checkbox(
                value=False,
                label="Try GPU (automatically falls back to CPU if it runs out of memory)",
            )
            run_btn = gr.Button("Separate", variant="primary")
            status = gr.HTML(value=_status_html("idle", "Idle"))
        with gr.Column(scale=1):
            stem_players = [
                gr.Audio(label=name.capitalize(), visible=False, interactive=False)
                for name in STEM_ORDER
            ]

    run_btn.click(
        fn=run_separation,
        inputs=[audio_in, preset, use_gpu],
        outputs=[status] + stem_players,
    )

    gr.Markdown(
        "---\n"
        "**Note:** this separates by instrument *category* -- e.g. one combined "
        '"guitar" stem, even if two guitars are playing. Splitting two simultaneous '
        "performers of the *same* instrument isn't something any current pretrained "
        "model can do reliably; see the Future Ideas section in README.md."
    )

if __name__ == "__main__":
    demo.launch()
