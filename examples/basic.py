from pre_reasoning import analyze_form, get_form

print(get_form()["template"])

# The calling AI writes this after interpreting the user's original context.
form = """DEPENDENCIES
Frontend depends on API.
API depends on Auth.
Auth depends on Key Management.
Key Management depends on HSM Provisioning.
HSM Provisioning depends on Procurement.
"""

analysis = analyze_form(form)
print(analysis["trace"])
