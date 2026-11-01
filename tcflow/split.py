"""Data splits for evaluation seeds (set TCFLOW_SPLIT). 'explore' = the y-sets inspected while
developing (2026-10-26..28); 'dev' = tuning grid selection; 'test' = confirmation set of round 1 (seen 29 Oct);
'test2' = confirmation set of round 2 (seen 30 Oct); 'test3' = confirmation under the amended gate metrics."""
import os

SPLIT = os.environ.get("TCFLOW_SPLIT", "explore")
OFF = {"explore": 0, "dev": 1_000_000, "test": 2_000_000, "test2": 3_000_000, "test3": 4_000_000}[SPLIT]
TAG = "" if SPLIT == "explore" else f"{SPLIT}_"  # cache-key / file prefix (explore keeps old names)
