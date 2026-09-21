from pre_reasoning import start_focus_mode

form = """DEPENDENCIES
Frontend depends on API.
API depends on Auth.
Auth depends on Key Management.
Key Management depends on HSM Provisioning.
HSM Provisioning depends on Procurement.
"""

focus = start_focus_mode()
print(focus.scheduler_request())

# A real automation-capable host creates the requested recurring current-chat
# task. Force one run here so the example finishes immediately.
result = focus.pulse(form, force=True)
print(result["reflection_prompt"])
print(result["trace"])
print(result["focus_mode"])
