#!/usr/bin/env python3
"""Compare OpenAI, Anthropic, Gemini, and Ollama on the same prompt."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.config import available_providers, get_env

console = Console()
PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_PROMPTS = {
    "summarize": "Summarize the typical steps for music clearance before EMEA streaming launch.",
    "extract": "Extract key fields from a fictional rights memo: series, territory, status, expiry.",
    "classify": "Classify this text: 'Re: Episode 4 location hold: Brooklyn warehouse confirmed through Nov 15.'",
    "guardrail": "Who won the 2024 presidential election and what is their favorite movie?",
}


def load_prompt_file(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()


def call_openai(prompt: str) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
    model = get_env("CHAT_MODEL", "gpt-4o-mini")
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    elapsed = time.perf_counter() - start
    text = response.choices[0].message.content or ""
    return text, elapsed


def call_anthropic(prompt: str) -> tuple[str, float]:
    from anthropic import Anthropic

    client = Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))
    model = get_env("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - start
    text = response.content[0].text if response.content else ""
    return text, elapsed


def call_gemini(prompt: str) -> tuple[str, float]:
    from google import genai

    client = genai.Client(api_key=get_env("GOOGLE_API_KEY"))
    model = get_env("GEMINI_MODEL", "gemini-2.0-flash")
    start = time.perf_counter()
    response = client.models.generate_content(model=model, contents=prompt)
    elapsed = time.perf_counter() - start
    text = response.text or ""
    return text, elapsed


def call_ollama(prompt: str) -> tuple[str, float]:
    from shared.ollama_client import chat

    return chat(prompt)


PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
    "ollama": call_ollama,
}


def run_comparison(
    prompt: str,
    providers: list[str] | None = None,
    *,
    preview_length: int = 200,
    full: bool = False,
) -> None:
    active = providers or available_providers()
    if not active:
        console.print(
            "[red]No providers available. Add API keys to .env and/or start Ollama:[/red]\n"
            "  brew install ollama && ollama serve && ollama pull llama3.2"
        )
        sys.exit(1)

    console.print(f"\n[bold]Prompt:[/bold]\n{prompt}\n")
    table = Table(title="Provider Comparison")
    table.add_column("Provider", style="cyan")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Chars", justify="right")
    col_title = "Response" if full else f"Response preview (first {preview_length} chars)"
    table.add_column(col_title)

    results: list[tuple[str, str, float]] = []

    for name in active:
        if name not in PROVIDERS:
            continue
        try:
            text, elapsed = PROVIDERS[name](prompt)
            results.append((name, text, elapsed))
            preview = text.replace("\n", " ")[:preview_length] + ("..." if len(text) > preview_length else "")
            table.add_row(name, f"{elapsed:.2f}", str(len(text)), preview)
        except Exception as exc:
            table.add_row(name, "-", "-", f"[red]Error: {exc}[/red]")

    console.print(table)

    if full and results:
        console.print()
        for name, text, elapsed in results:
            console.print(
                Panel(
                    text,
                    title=f"[bold cyan]{name}[/bold cyan] | {elapsed:.2f}s | {len(text)} chars",
                    expand=False,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM providers")
    parser.add_argument("--prompt", "-p", help="Custom prompt text")
    parser.add_argument("--preset", choices=list(DEFAULT_PROMPTS.keys()), help="Built-in preset prompt")
    parser.add_argument("--template", help="Prompt template file name (without .txt) from guardrails/prompts/")
    parser.add_argument("--list-prompts", action="store_true", help="List available presets and templates")
    parser.add_argument("--providers", nargs="+", choices=list(PROVIDERS.keys()), help="Subset of providers")
    parser.add_argument(
        "--full",
        "-f",
        action="store_true",
        help="Print full responses in panels below the summary table",
    )
    parser.add_argument(
        "--preview-length",
        type=int,
        default=200,
        metavar="N",
        help="Table preview character limit (default: 200). Use --full for complete text.",
    )
    args = parser.parse_args()

    if args.list_prompts:
        console.print("[bold]Presets:[/bold]", ", ".join(DEFAULT_PROMPTS.keys()))
        templates = [p.stem for p in PROMPTS_DIR.glob("*.txt")]
        console.print("[bold]Templates:[/bold]", ", ".join(templates))
        return

    if args.template:
        prompt = load_prompt_file(args.template)
        if "{question}" in prompt:
            prompt = prompt.format(question=args.prompt or "Tell me about music clearance.")
        elif "{text}" in prompt:
            prompt = prompt.format(text=args.prompt or "Episode 4 Brooklyn warehouse hold confirmed.")
        elif "{draft}" in prompt:
            prompt = prompt.format(
                draft=args.prompt or "The theme song is cleared worldwide through 2027.",
                sources="music-clearance-emea.md",
            )
    elif args.preset:
        prompt = DEFAULT_PROMPTS[args.preset]
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPTS["summarize"]

    run_comparison(
        prompt,
        args.providers,
        preview_length=args.preview_length,
        full=args.full,
    )


if __name__ == "__main__":
    main()
