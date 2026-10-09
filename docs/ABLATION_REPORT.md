# Historical retrieval diagnostics

This report describes the five-query run in `data/ablation_results.json`. It is exploratory evidence, not a labeled accuracy benchmark. Original raw results are preserved for traceability.

- Reported latency: about 0.60 s for flat retrieval and 0.33 s for hierarchical retrieval. A single ordered run cannot isolate cache, network or warm-up effects; no general 45% speedup is established.
- Graph retrieval returned results for 2 of 5 queries. Returning results is not a correctness measure.
- Internal mean scores of approximately 0.664 and 12.8 use different formulas. Graph scores include entity/path bonuses. Their ratio is not a 19× accuracy improvement.
- The historical hybrid branch reused hierarchical retrieval. The current script reports it as unimplemented rather than presenting a duplicate as a separate experiment.

Use shared relevance labels and the [evaluation protocol](EVALUATION_PROTOCOL.md) before making comparative claims. No replacement end-to-end experiment has been run as part of this correction.
