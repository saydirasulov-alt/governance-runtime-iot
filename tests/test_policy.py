import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.policy_loader import load_policy, evaluate_policy, set_rule_enabled
import copy

BASE = "configs/policy_gates.yaml"

def _i(**kw):
    d = dict(setpoint=22.0, action="set_temperature", timestamp=1,
             source="ai", intent_id="i", device_id="d", queue_depth=0)
    d.update(kw); return d

def test_safe_intent_admitted():
    assert evaluate_policy(load_policy(BASE), _i())[0] == "PASS"

def test_g1_setpoint_out_of_band_rejected():
    assert evaluate_policy(load_policy(BASE), _i(setpoint=41.0)) == ("REJECT", "G1")

def test_g1_disallowed_action_rejected():
    assert evaluate_policy(load_policy(BASE), _i(action="reboot_grid")) == ("REJECT", "G1")

def test_g2_missing_timestamp_rejected():
    assert evaluate_policy(load_policy(BASE), _i(timestamp=None)) == ("REJECT", "G2")

def test_g3_queue_overload_throttled():
    assert evaluate_policy(load_policy(BASE), _i(queue_depth=150)) == ("THROTTLE", "G3")

def test_g4_missing_identifier_rejected():
    assert evaluate_policy(load_policy(BASE), _i(intent_id=None)) == ("REJECT", "G4")

def test_known_g2_gap_forged_provenance_is_admitted():
    """The shipped policy admits a forged-provenance intent: the documented G2 gap."""
    assert evaluate_policy(load_policy(BASE), _i(source=None))[0] == "PASS"

def test_corrected_policy_closes_the_gap():
    p = set_rule_enabled(copy.deepcopy(load_policy(BASE)), "G2", "source", True)
    assert evaluate_policy(p, _i(source=None)) == ("REJECT", "G2")
