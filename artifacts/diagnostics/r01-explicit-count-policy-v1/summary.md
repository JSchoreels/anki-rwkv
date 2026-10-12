# Explicit-count R01 policy clarification

Human instruction (2026-10-12): "Perfect, but if I ask for a certain amount of round, don't stop early". Future explicitly counted rounds complete their requested count despite speed misses. Uncounted rounds retain the default three-miss stop. Historical round 0041–0045 protocols and results are unchanged; no continuation was launched.

`--round-authorization` plus `--round-iterations N` binds the human count in immutable protocols. Later preparation carries budget/index; selection and both launch guards honor the count, keep miss reporting/acceptance gates, and reject N+1 without new authorization. Partial summary/handoff records retain the remaining authorized iterations.

CPU-only evidence: [116 policy checks](policy/result.json) cover nine missed iterations, exhaustion, malformed/missing authorization, new uncounted rounds and preparation propagation; [33 helper tests](helpers-handoff/result.json) include partial/count-complete handoff behavior. The earlier [32-test helper check](helpers/result.json) predates the handoff addition and remains preserved. One CPU preflight was reused; source/runtime/provenance metadata and logs remain alongside each result. No training, CUDA diagnostic or broader validation ran; performance effects remain unmeasured.

Implementation commit after checks: `7795eb04ccf1869fb4df7d0b5f7e2d0465fd992f`. This later revision is not substituted for the dirty experiment/check revision in retained results.
