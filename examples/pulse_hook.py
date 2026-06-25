from pre_reasoning import analyze, pulse


problem = "Frontend depends on API. API depends on Auth."
draft = "Fix Auth first, then verify API before frontend work."

print(analyze(problem)["trace"])
print(pulse(problem, draft))
