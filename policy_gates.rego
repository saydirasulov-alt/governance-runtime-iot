# OPA Rego Policy - Governance Gates G1-G4
# Manuscript: sensors-4349708
# Rego v1 syntax (OPA v1.0+); semantics identical to the paper's G1-G4 gates.

package governance
import rego.v1

default allow := false

# G1: Safety - setpoint bounds and action validity
g1_pass if {
	input.setpoint >= 15.0
	input.setpoint <= 30.0
	input.action in {"set_temperature", "set_mode", "set_fan"}
}

# G2: Privacy - metadata integrity
# NOTE: checks timestamp only, NOT source field
# This is the specification gap identified in Section 5.4
g2_pass if {
	input.timestamp != null
}

# G3: Resilience - queue health
g3_pass if {
	input.queue_depth < 100
}

# G4: Auditability
g4_pass if {
	input.intent_id != null
	input.device_id != null
}

allow if {
	g1_pass
	g2_pass
	g3_pass
	g4_pass
}
