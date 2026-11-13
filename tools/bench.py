## File Name: bench.py
## Creator: DatJavaClass
## Associated Files: rt_net.py, rt_turbo.py, range_server.py
## Exposed Vars: None
## Exposed Funcs: main
## Desc: Compare capped local download throughput.
import argparse
import math
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
import rt_net
import rt_turbo
from range_server import RangeServer

def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0: ## invalid input exits through argparse
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number

def main():
    parser = argparse.ArgumentParser(description="Benchmark local capped downloads; MB means 1048576 bytes.")
    parser.add_argument("--size-mb", type=positive, default=64)
    parser.add_argument("--rate-mbps", type=positive, default=4, help="MB/s per connection")
    args = parser.parse_args()
    payload = b"x" * max(1, int(args.size_mb * 1048576))
    server = RangeServer({"bench.bin": payload}, rate=args.rate_mbps * 1048576)
    try:
        url = server.start() + "/f/bench.bin"
        with tempfile.TemporaryDirectory(prefix="rt_bench_") as folder:
            print(f"{'Method':<12} {'Connections':>11} {'Seconds':>10} {'MB/s':>10}")
            for connections in (0, 1, 4, 8, 16): ## exits after all five measurements
                dest = os.path.join(folder, f"bench-{connections}.bin")
                started = time.perf_counter()
                result = rt_turbo.fetch(url, dest, connections=connections) if connections else rt_net.http_download(url, dest)
                elapsed = time.perf_counter() - started
                if not result[0] or not result[1]: ## unsuccessful download aborts benchmark
                    raise RuntimeError("benchmark download did not finish")
                with open(result[1], "rb") as stream:
                    if stream.read() != payload: ## invalid payload aborts benchmark
                        raise RuntimeError("benchmark payload differs")
                method = "V3 fetch" if connections else "V2 stream"
                print(f"{method:<12} {connections or 1:>11} {elapsed:>10.3f} {len(payload) / 1048576 / elapsed:>10.2f}")
    finally:
        server.stop()

if __name__ == "__main__": ## direct execution enters benchmark
    main()
