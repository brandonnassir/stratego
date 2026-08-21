"""Kill a loader worker mid-epoch, with many minibatches still pending."""
import os, sys, time, threading, tempfile, subprocess, traceback
from pathlib import Path


def watchdog(pid, delay_after_pool):
    """Wait for the pool, let it get going, then kill one worker."""
    while True:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
        workers = []
        for token in out:
            cmd = subprocess.run(["ps", "-o", "command=", "-p", token],
                                 capture_output=True, text=True).stdout
            if "spawn_main" in cmd:
                workers.append(int(token))
        if len(workers) >= 2:
            time.sleep(delay_after_pool)
            still = []
            for w in workers:
                st = subprocess.run(["ps","-o","state=","-p",str(w)],capture_output=True,text=True).stdout.strip()
                if st and not st.startswith("Z"):
                    still.append(w)
            if still:
                victim = still[0]
                print(f"WATCHDOG: killing loader worker {victim} of {still}", flush=True)
                os.kill(victim, 9)
            return
        time.sleep(0.2)


def main():
    from stratego.training.phase14_contract import Population
    from stratego.training.phase14_runner import MODE_TEST, Phase14Runner
    from stratego.training.phase14_storage import Phase14Storage
    from stratego.training.phase14_clock import ManualClock
    from stratego.training.phase9_trainer import LoaderTopology

    root = Path(tempfile.mkdtemp(prefix="phase13_workerkill_"))
    r = Phase14Runner(Phase14Storage.under(root), clock=ManualClock("2026-09-01T00:00:00.000Z"),
                      mode=MODE_TEST, device="mps", inference_device="mps",
                      topology=LoaderTopology(workers=6), games_in_flight=96,
                      population=Population.scaled(8))
    r.start()
    threading.Thread(target=watchdog, args=(os.getpid(), 3.0), daemon=True).start()
    try:
        unit = r.run_iteration()
        print("ITERATION SURVIVED:", {k: unit.get(k) for k in ("iteration","sealed","trained","updates")}, flush=True)
    except BaseException as e:
        print("ITERATION RAISED:", type(e).__module__ + "." + type(e).__name__, str(e)[:200], flush=True)
        print("MRO:", [c.__name__ for c in type(e).__mro__], flush=True)
        traceback.print_exc()
        subprocess.run(["rm","-rf",str(root)])
        return 1
    subprocess.run(["rm","-rf",str(root)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
