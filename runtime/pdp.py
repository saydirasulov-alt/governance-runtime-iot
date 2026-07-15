"""Policy Decision Points. Three real backends; no simulated timing anywhere.

inline : in-process evaluation of the declarative policy
http   : a real HTTP policy service (policy_http_server.py)
opa    : a REAL Open Policy Agent server (REST /v1/data/governance/decision)
"""
import json, urllib.request
from .policy_loader import load_policy, evaluate_policy


class InlinePDP:
    name = "inline"

    def __init__(self, policy_path=None):
        self.policy = load_policy(policy_path)

    def decide(self, intent):
        return evaluate_policy(self.policy, intent)


class HttpPDP:
    name = "http"

    def __init__(self, url="http://127.0.0.1:8181/decide", timeout=2.0):
        self.url, self.timeout = url, timeout

    def decide(self, intent):
        req = urllib.request.Request(
            self.url, data=json.dumps(intent, default=str).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
        return r.get("decision", "REJECT"), r.get("gate")


class OpaPDP:
    """Real OPA. Expects policy/governance.rego loaded into an OPA server."""
    name = "opa"

    def __init__(self, url="http://127.0.0.1:8181/v1/data/governance/decision", timeout=2.0):
        self.url, self.timeout = url, timeout

    def decide(self, intent):
        body = json.dumps({"input": intent}, default=str).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
        res = r.get("result") or {}
        return res.get("outcome", "REJECT"), res.get("gate")


def make_pdp(kind, **kw):
    return {"inline": InlinePDP, "http": HttpPDP, "opa": OpaPDP}[kind](**kw)
