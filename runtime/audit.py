"""Persistent, tamper-evident hash-chained audit log (append-only JSONL)."""
import json, hashlib, os, threading

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path="results/audit.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self.last_hash = self._recover_tail()

    def _recover_tail(self):
        if not os.path.exists(self.path):
            return GENESIS
        last = GENESIS
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)["hash"]
        return last

    def append(self, event, detail):
        with self._lock:
            rec = {"seq": self._next_seq(), "event": event, "detail": detail,
                   "prev": self.last_hash}
            h = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
            rec["hash"] = h
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.last_hash = h
            return rec

    def _next_seq(self):
        if not hasattr(self, "_seq"):
            self._seq = sum(1 for _ in open(self.path)) if os.path.exists(self.path) else 0
        self._seq += 1
        return self._seq

    def verify(self):
        prev = GENESIS
        n = 0
        if not os.path.exists(self.path):
            return True, 0
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["prev"] != prev:
                    return False, n
                c = dict(rec); h = c.pop("hash")
                if hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest() != h:
                    return False, n
                prev = h; n += 1
        return True, n
