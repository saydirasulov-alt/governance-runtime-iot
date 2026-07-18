# Real OPA policy: the same G1-G4 gates, evaluated by a real OPA server.
# Load with:  opa run --server --addr :8181 policy/governance.rego
# Query:      POST /v1/data/governance/decision   {"input": <intent>}
package governance

default decision := {"outcome": "REJECT", "gate": "G4"}

decision := {"outcome": "REJECT", "gate": "G1"} if {
	input.setpoint != null
	not within_band
}
decision := {"outcome": "REJECT", "gate": "G1"} if {
	not allowed_action
}
decision := {"outcome": "REJECT", "gate": "G2"} if {
	within_band; allowed_action
	input.timestamp == null
}
decision := {"outcome": "THROTTLE", "gate": "G3"} if {
	within_band; allowed_action; input.timestamp != null
	input.queue_depth >= 100
}
decision := {"outcome": "PASS", "gate": null} if {
	within_band; allowed_action
	input.timestamp != null
	input.queue_depth < 100
	input.intent_id != null
	input.device_id != null
}

within_band if { input.setpoint >= 15.0; input.setpoint <= 30.0 }
within_band if { input.setpoint == null }
allowed_action if { input.action in {"set_temperature", "set_mode", "set_fan"} }
