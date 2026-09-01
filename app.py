"""
Stem Splitter GUI.

A small Gradio app: drop in a track, pick a separation mode, get stems back
as playable/downloadable audio. Separation itself is done by pretrained
Demucs models (see backend.py) -- nothing here is trained from scratch.
"""
import queue
import threading
from pathlib import Path

import gradio as gr

from backend import DEFAULT_PRESET, MODEL_PRESETS, STEM_ORDER, separate, split_by_stereo_position

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

    # status, *stem_players, stems_state, stem_choice
    skip_stems = (gr.skip(),) * (len(STEM_ORDER) + 2)

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

    stems_as_str = {name: str(path) for name, path in result.stems.items()}
    available = [name for name in STEM_ORDER if name in result.stems]
    default_choice = "guitar" if "guitar" in available else available[0]
    stem_choice_update = gr.update(choices=available, value=default_choice, interactive=True)

    yield (_status_html("done", "Done"),) + tuple(stem_updates) + (stems_as_str, stem_choice_update)


def run_position_split(stems, stem_name, strength):
    if not stems or stem_name not in stems:
        raise gr.Error("Run a separation first, then pick a stem to split.")
    stem_path = stems[stem_name]
    # Save alongside the stem it came from (output/<track name>/...) rather
    # than loose in output/, matching how the main separation is organized.
    result = split_by_stereo_position(stem_path, strength=strength, output_dir=str(Path(stem_path).parent))
    return (
        gr.update(value=str(result["left"]), visible=True, label=f"{stem_name.capitalize()} — left-leaning"),
        gr.update(value=str(result["right"]), visible=True, label=f"{stem_name.capitalize()} — right-leaning"),
    )


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

    stems_state = gr.State(value={})  # stem name -> file path, from the last separation

    gr.Markdown(
        "---\n"
        "**Note:** the stems above separate by instrument *category* -- e.g. one "
        'combined "guitar" stem, even if two guitars are playing. Full source '
        "identification for multiple performers of the *same* instrument isn't "
        "something any current pretrained model can do reliably."
    )

    with gr.Accordion("🧪 Split a stem by stereo position (beta)", open=False):
        gr.Markdown(
            "Experimental, not true source separation. Splits a stem into a "
            "left-leaning and right-leaning pair based purely on stereo panning "
            "-- useful when two performances of the same instrument (e.g. "
            "double-tracked guitars) were panned apart in the mix. **Does "
            "nothing useful if the source is centered/mono** -- there's no "
            "stereo cue to exploit in that case, and no algorithm can invent one."
        )
        with gr.Row():
            split_stem_choice = gr.Dropdown(choices=[], label="Stem to split", interactive=False)
            split_strength = gr.Slider(
                minimum=0.0, maximum=0.6, value=0.3, step=0.05,
                label="Separation strength",
                info="0 = plain left/right channels (best for hard-panned sources). "
                     "Higher also cancels shared/centered content (better for moderately-panned sources).",
            )
        split_btn = gr.Button("Split by position")
        with gr.Row():
            left_player = gr.Audio(label="Left-leaning", visible=False, interactive=False)
            right_player = gr.Audio(label="Right-leaning", visible=False, interactive=False)

        split_btn.click(
            fn=run_position_split,
            inputs=[stems_state, split_stem_choice, split_strength],
            outputs=[left_player, right_player],
        )

    run_btn.click(
        fn=run_separation,
        inputs=[audio_in, preset, use_gpu],
        outputs=[status] + stem_players + [stems_state, split_stem_choice],
    )

if __name__ == "__main__":
    demo.launch()
