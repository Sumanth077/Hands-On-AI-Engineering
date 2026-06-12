"""Gradio UI for the Multilingual Audio Translator."""

from __future__ import annotations

import gradio as gr

from synthesizer import synthesize
from transcriber import transcribe
from translator import LANGUAGE_NAMES, translate

TARGET_LANGUAGES = list(LANGUAGE_NAMES.items())


def process_audio(audio_path: str | None, target_language: str):
    """Run the full transcribe, translate, and synthesize pipeline and return all outputs for the UI."""
    if not audio_path:
        raise gr.Error("Please upload or record an audio file.")

    if not target_language:
        raise gr.Error("Please select a target language.")

    transcription = transcribe(audio_path)
    translated_text = translate(
        transcription.text,
        transcription.language,
        target_language,
    )
    output_audio = synthesize(translated_text, target_language)

    source_label = LANGUAGE_NAMES.get(
        transcription.language,
        transcription.language,
    )
    target_label = LANGUAGE_NAMES.get(target_language, target_language)

    source_info = (
        f"**Detected language:** {source_label} ({transcription.language})  \n"
        f"**Confidence:** {transcription.language_probability:.1%}"
    )
    status = (
        f"Translated from {source_label} to {target_label}. "
        "Play the synthesized audio below."
    )

    return (
        source_info,
        transcription.text,
        translated_text,
        output_audio,
        status,
    )


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface for the audio translator."""
    with gr.Blocks(
        title="Multilingual Audio Translator",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Multilingual Audio Translator

            Upload or record speech in any language. The app transcribes it with
            **faster-whisper**, translates the text with an LLM via **Orq.ai**, and
            synthesizes the translation with **Kokoro TTS**.
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Input audio (upload MP3/WAV or record)",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                target_language = gr.Dropdown(
                    label="Target language",
                    choices=TARGET_LANGUAGES,
                    value="en",
                )
                translate_btn = gr.Button("Translate", variant="primary")

            with gr.Column():
                status = gr.Markdown("")
                source_info = gr.Markdown(label="Source language")
                original_text = gr.Textbox(
                    label="Original transcript",
                    lines=4,
                    interactive=False,
                )
                translated_text = gr.Textbox(
                    label="Translated text",
                    lines=4,
                    interactive=False,
                )
                output_audio = gr.Audio(
                    label="Translated speech",
                    type="filepath",
                    interactive=False,
                )

        translate_btn.click(
            fn=process_audio,
            inputs=[audio_input, target_language],
            outputs=[
                source_info,
                original_text,
                translated_text,
                output_audio,
                status,
            ],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
