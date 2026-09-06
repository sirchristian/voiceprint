from pathlib import Path

from voiceprint.gguf_export import build_llama_cpp_command


def test_build_llama_cpp_command_uses_expected_paths() -> None:
    cmd = build_llama_cpp_command(
        model_dir=Path("models/user-voice-merged"),
        output_dir=Path("models/user-voice-gguf"),
        outfile="user-voice-f16.gguf",
        outtype="f16",
        llama_cpp_dir=Path("/opt/llama.cpp"),
    )

    assert "convert_hf_to_gguf.py" in cmd
    assert "models/user-voice-merged" in cmd
    assert "models/user-voice-gguf/user-voice-f16.gguf" in cmd
    assert "--outtype f16" in cmd
