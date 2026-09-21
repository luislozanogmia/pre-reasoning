"""Console entry point for Pre-Reasoning V4."""
import argparse
import json

from . import __version__, analyze_form, get_engine, get_form


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an AI-authored structured pre-reasoning form."
    )
    parser.add_argument("form_text", nargs="?")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--form", action="store_true")
    args = parser.parse_args()
    if args.form:
        print(json.dumps(get_form(), indent=2))
        return
    if args.info:
        engine = get_engine()
        print(json.dumps({
            "version": __version__,
            "model": engine.model_meta["variant_id"],
            "params": engine.params,
            "device": engine.device,
            "strict_checkpoint_load": engine.model_meta["strict_load"],
        }, indent=2))
        return
    if not args.form_text:
        parser.error("structured form text is required; run --form for the contract")
    result = analyze_form(args.form_text)
    print(json.dumps(result, indent=2) if args.json else result["trace"])

if __name__ == "__main__":
    main()
