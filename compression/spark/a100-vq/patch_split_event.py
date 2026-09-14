"""Let a load's H2D copy overlap the resident pass instead of queueing behind it.

`CB3Store.load` does `stream.wait_stream(compute)` so a record never lands on a slot some running
kernel is still reading. Under resident-first that is too strong: the copy then waits for the
resident MoE that was just launched, every loader thread blocks on it, and the read queue drains --
measured as the disk sitting idle a quarter of the time.

What the copy actually has to be ordered after is everything up to the PREVIOUS layer, because the
slots it writes were allocated in this call and no kernel launched before it can be reading them.
So `resolve` records an event at that moment and the store waits on the event instead.
"""
import os
import shutil

W = os.path.expanduser("~/dsv41-spark/work")

p = f"{W}/engine/experts.py"
s = open(p).read()
a = """            t0 = time.perf_counter()
            futs = [self.pool.submit(self._load_into_slot, *ks) for ks in to_load]"""
b = """            t0 = time.perf_counter()
            # everything queued so far -- i.e. through the previous layer. The slots below were
            # allocated in this call, so nothing launched before this point can be reading them,
            # and the copies may run alongside the resident pass we are about to launch.
            import cb3_store as _cs
            ev = torch.cuda.Event()
            ev.record()
            _cs.GATE = ev
            futs = [self.pool.submit(self._load_into_slot, *ks) for ks in to_load]"""
assert a in s, "split submit not found"
open(p, "w").write(s.replace(a, b, 1))
print("patched engine/experts.py: gate event recorded before the resident pass")

p = os.path.expanduser("~/dsv41-spark/a100-vq/cb3_store.py")
bak = p + ".pre-event.bak"
if not os.path.exists(bak):
    shutil.copy2(p, bak)
s = open(bak).read()
a = """        compute = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            stream.wait_stream(compute)"""
b = """        compute = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            # GATE, when the engine's split path set one, is an event recorded before the resident
            # MoE was launched: ordering the copy after THAT instead of after the whole compute
            # stream is what lets the read queue stay full while the resident pass runs.
            gate = GATE
            if gate is None:
                stream.wait_stream(compute)
            else:
                stream.wait_event(gate)"""
assert a in s, "wait_stream not found"
s = s.replace(a, b, 1)
s = s.replace("PROBE = _Probe() if os.environ.get(\"DSV41_STORE_PROBE\") == \"1\" else None",
              "PROBE = _Probe() if os.environ.get(\"DSV41_STORE_PROBE\") == \"1\" else None\n"
              "#: set by ExpertStore.resolve's split path to an event recorded before the resident pass\n"
              "GATE = None", 1)
open(p, "w").write(s)
print("patched a100-vq/cb3_store.py: copies wait on the gate event")
