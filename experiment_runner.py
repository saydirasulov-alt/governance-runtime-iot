"""Experiment runner — 6 real experiments x 30 seeds."""
import json, time, random, csv, os
from governance_runtime import GovernanceRuntime

CONFIG = json.load(open("experiment_config.json"))
os.makedirs("results", exist_ok=True)

FIELDS = ["seed","experiment","variant","device","fault","backend","queue_depth",
          "num_devices","t_submit","t_policy_start","t_policy_end","t_execute",
          "t_commit","outcome","reason","ground_truth_unsafe","policy_ms",
          "e2e_ms","sim_e2e_ms","sim_mttr_ms","rollback_success"]

def gen_intent(did, fault=None):
    b = {"setpoint":22.0, "action":"set_temperature", "timestamp":time.time(),
         "source":f"s_{did}", "intent_id":f"i_{did}_{random.randint(0,9999)}",
         "device_id":f"d_{did}"}
    if fault == "unsafe_setpoint": b["setpoint"] = 14.9
    elif fault == "corrupted_meta": b["timestamp"] = None
    elif fault == "boundary_setpoint": b["setpoint"] = 14.99
    elif fault == "partial_corruption": b["source"] = None
    elif fault == "stealthy_unsafe": b["setpoint"] = 14.5
    elif fault == "multi_vector": b["setpoint"] = 14.9; b["source"] = None
    return b

def run(rt, seed, nd, fr, exp, var=""):
    random.seed(seed)
    traces = []
    faults = ["unsafe_setpoint","corrupted_meta","queue_flood","boundary_setpoint",
              "partial_corruption","stealthy_unsafe","multi_vector"]
    for i in range(nd):
        t = {k: None for k in FIELDS}
        t["seed"]=seed; t["experiment"]=exp; t["variant"]=var; t["device"]=i
        t["num_devices"]=nd; t["backend"]=rt.backend
        fault = random.choice(faults) if random.random() < fr else None
        t["fault"] = fault
        # Queue depth: queue_flood=150, partial_corruption=150 (80%) or normal (20%)
        if fault == "queue_flood":
            q = 150
        elif fault == "partial_corruption":
            q = 150 if random.random() < 0.8 else random.randint(0, 50)
        else:
            q = random.randint(0, 50)
        t["queue_depth"] = q
        r = rt.admit(gen_intent(i, fault), queue_depth=q, num_devices=nd)
        for k in ["t_submit","t_policy_start","t_policy_end","t_execute","t_commit",
                   "outcome","reason","ground_truth_unsafe","sim_e2e_ms","sim_mttr_ms","rollback_success"]:
            t[k] = r.get(k)
        if t["t_policy_start"] and t["t_policy_end"]:
            t["policy_ms"] = (t["t_policy_end"]-t["t_policy_start"])*1000
        if t["t_submit"] and t["t_commit"]:
            t["e2e_ms"] = (t["t_commit"]-t["t_submit"])*1000
        traces.append(t)
    return traces

def save(traces, fn):
    with open(f"results/{fn}", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        csv.DictWriter(f, fieldnames=FIELDS).writerows(traces)
    print(f"  {fn}: {len(traces)} rows")

def exp1():
    print("Exp 1: Backend comparison")
    a = []
    for s in CONFIG["seeds"]:
        for backend in ["inline", "http_rest", "subprocess_cli"]:
            a.extend(run(GovernanceRuntime(backend=backend), s, 200, 0.1, "backend", backend))
    save(a, "01_backend_comparison.csv")

def exp2():
    print("Exp 2: Ablation")
    a = []
    v = {"GA-RT":{}, "NoPolicy":{"enable_policy":False}, "NoRollback":{"enable_rollback":False},
         "NoCheckpoint":{"enable_checkpoint":False}, "NoAudit":{"enable_audit":False}}
    for s in CONFIG["seeds"]:
        for n, f in v.items():
            a.extend(run(GovernanceRuntime(**f), s, 200, 0.1, "ablation", n))
    save(a, "02_ablation.csv")

def exp3():
    print("Exp 3: Baseline")
    a = []
    for s in CONFIG["seeds"]:
        a.extend(run(GovernanceRuntime(), s, 200, 0.1, "baseline", "GA-RT"))
        a.extend(run(GovernanceRuntime(cloud_delay_ms=10.0), s, 200, 0.1, "baseline", "CC"))
        a.extend(run(GovernanceRuntime(enable_policy=False, enable_rollback=False,
                     cloud_delay_ms=3.0), s, 200, 0.1, "baseline", "MS-NoGov"))
    save(a, "03_baseline.csv")

def exp4():
    print("Exp 4: Network sensitivity")
    a = []
    for s in CONFIG["seeds"]:
        for loss, jit in [(0,0),(0.05,5),(0.10,10),(0.20,20)]:
            a.extend(run(GovernanceRuntime(network_loss=loss, network_jitter_ms=jit),
                        s, 200, 0.1, "network", f"loss{int(loss*100)}_jit{jit}"))
    save(a, "04_network_sensitivity.csv")

def exp5():
    print("Exp 5: Adversarial")
    a = []
    for s in CONFIG["seeds"]:
        a.extend(run(GovernanceRuntime(), s, 200, 0.2, "adversarial", "default"))
    save(a, "05_adversarial.csv")


def run_scale(rt, seed, full_n, fr, var):
    """Scalability: iterate min(n,100) devices but use full_n for queue model."""
    random.seed(seed)
    traces = []
    faults = ["unsafe_setpoint","corrupted_meta","queue_flood","boundary_setpoint",
              "partial_corruption","stealthy_unsafe","multi_vector"]
    nd = min(full_n, 100)
    for i in range(nd):
        t = {k: None for k in FIELDS}
        t["seed"]=seed; t["experiment"]="scalability"; t["variant"]=var; t["device"]=i
        t["num_devices"]=full_n; t["backend"]=rt.backend
        fault = random.choice(faults) if random.random() < fr else None
        t["fault"] = fault
        if fault == "queue_flood":
            q = 150
        elif fault == "partial_corruption":
            q = 150 if random.random() < 0.8 else random.randint(0, 50)
        else:
            q = random.randint(0, 50)
        t["queue_depth"] = q
        r = rt.admit(gen_intent(i, fault), queue_depth=q, num_devices=full_n)
        for k in ["t_submit","t_policy_start","t_policy_end","t_execute","t_commit",
                   "outcome","reason","ground_truth_unsafe","sim_e2e_ms","sim_mttr_ms","rollback_success"]:
            t[k] = r.get(k)
        if t["t_policy_start"] and t["t_policy_end"]:
            t["policy_ms"] = (t["t_policy_end"]-t["t_policy_start"])*1000
        if t["t_submit"] and t["t_commit"]:
            t["e2e_ms"] = (t["t_commit"]-t["t_submit"])*1000
        traces.append(t)
    return traces

def exp6():
    print("Exp 6: Scalability")
    a = []
    for s in CONFIG["seeds"]:
        for n in CONFIG["num_devices"]:
            a.extend(run_scale(GovernanceRuntime(), s, n, 0.1, f"n{n}"))
    save(a, "06_scalability.csv")

if __name__ == "__main__":
    print("Running 6 experiments...")
    exp1(); exp2(); exp3(); exp4(); exp5(); exp6()
    print("Done.")
