# OPA Rego Policy - Governance Gates G1-G4
# Manuscript: sensors-4349708

package governance

default allow = false

# G1: Safety - setpoint bounds and action validity
g1_pass {
    input.setpoint >= 15.0
    input.setpoint <= 30.0
    allowed_actions := {"set_temperature", "set_mode", "set_fan"}
    allowed_actions[input.action]
}

# G2: Privacy - metadata integrity
# NOTE: checks timestamp only, NOT source field
# This is the specification gap identified in Section 5.4
g2_pass {
    input.timestamp != null
}

# G3: Resilience - queue health
g3_pass {
    input.queue_depth < 100
}

# G4: Auditability
g4_pass {
    input.intent_id != null
    input.device_id != null
}

allow {
    g1_pass
    g2_pass
    g3_pass
    g4_pass
}
