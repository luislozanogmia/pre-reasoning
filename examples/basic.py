from pre_reasoning import analyze


result = analyze("Frontend depends on API. API depends on Auth.")
print(result["trace"])
print(result["derived_assumptions"])
