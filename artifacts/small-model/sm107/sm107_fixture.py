"""The Big size on the show-time big model (SM107): the fork's bundled fixture rslib/src/rwkv/testdata/legacy-vectors.json.gz
= the first 500 rows per user of the full show-time vectors (sm107_vectors.py ... 0 e2s-query 4676 1745), gzip mtime 0.

usage: python sm107_fixture.py <full vectors .json> <out .json.gz>
"""
import gzip
import json
import sys
from pathlib import Path

users = json.loads(Path(sys.argv[1]).read_text())
for user in users:
    user["rows"] = user["rows"][:500]
Path(sys.argv[2]).write_bytes(gzip.compress(json.dumps(users, separators=(",", ":")).encode(), mtime=0))
print(sys.argv[2], sum(len(u["rows"]) for u in users), "rows", Path(sys.argv[2]).stat().st_size, "bytes")
