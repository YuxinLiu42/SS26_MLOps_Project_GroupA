# Load test (M24)

`tests/load/locustfile.py` run against the Cloud Run service
(`https://paligemma-api-...europe-west4.run.app`), 2 users, 4 minutes:

```bash
uv run --group serving locust -f tests/load/locustfile.py \
  --headless -u 2 -r 1 -t 4m --host <cloud-run-url> --csv reports/load/scienceqa
```

## Findings

| Endpoint | Requests | Failures | Median | p95 | Max |
|---|---|---|---|---|---|
| POST /predict | 24 | 13 (429) | 10 s | 27 s | 27 s |
| GET / | 12 | 8 (429) | 4.6 s | 10 s | 10 s |

- **CPU inference dominates**: successful `/predict` calls take ~10–27 s
  (PaliGemma2-3B on CPU, no GPU). Cold start is ~160 s (model download + load).
- **429 "Rate exceeded" under concurrency**: this run was against the initial
  deploy (`--max-instances 1 --concurrency 1`), so the service served one
  request at a time and rejected the overflow — and a health check colliding
  with an in-flight `/predict` was also 429'd. We then raised the service to
  `--max-instances 3 --concurrency 1`: overflow now spins a new instance
  instead of 429-ing, while still one heavy inference per instance (avoids
  OOM on the 3B model) and scale-to-zero when idle. Re-running the load test
  against the new config should show fewer 429s (at the cost of more cold
  starts, since each new instance downloads the base model on its first call).

Raw stats in `scienceqa_stats.csv` / `scienceqa_stats_history.csv`.
