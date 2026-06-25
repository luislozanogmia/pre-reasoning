from pre_reasoning import analyze


result = analyze(
    "CTO conflicts with senior dev. Release requires 80 percent test coverage.",
)

print(result["trace"])
