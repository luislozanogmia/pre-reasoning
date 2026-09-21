from pre_reasoning import analyze_form

# One AI-authored form can contain all five reasoning families.
form = """DEPENDENCIES
Production Launch depends on Security Review.
Security Review depends on Architecture Approval.

CONFLICTS
Fast Rollout conflicts with Safety Review.

REQUIREMENTS
Test Coverage must be at least 95.

CONDITIONALS
If Security Review passes, then Production Launch can proceed, otherwise Remediation must proceed.
"""

analysis = analyze_form(form)
print(analysis["trace"])
