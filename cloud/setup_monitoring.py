"""Set up Cloud Monitoring alerting for the Cloud Run API (M28).

Cloud Run already streams system metrics (request count, latency, container
CPU/memory utilization, instance count) to Cloud Monitoring with no code. This
script adds the alerting piece: an email notification channel and an alert
policy that fires on 5xx server errors from the `paligemma-api` service.

Idempotent by display name — safe to re-run. Run:
    python cloud/setup_monitoring.py
(uses Application Default Credentials; needs roles/monitoring.editor)
"""

import json
import os
import subprocess
import urllib.request

PROJECT = os.environ.get("GCP_PROJECT", "paligemma-scienceqa")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "liuyuxinnj@gmail.com")
SERVICE = os.environ.get("SERVICE_NAME", "paligemma-api")
BASE = f"https://monitoring.googleapis.com/v3/projects/{PROJECT}"

_TOKEN = subprocess.check_output(
    ["gcloud", "auth", "print-access-token"], text=True
).strip()
_HEADERS = {"Authorization": f"Bearer {_TOKEN}", "Content-Type": "application/json"}


def _api(path: str, body: dict | None = None) -> dict:
    """GET (body=None) or POST the Monitoring v3 API and return the JSON."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE}/{path}", data=data, headers=_HEADERS, method="POST" if body else "GET"
    )
    return json.load(urllib.request.urlopen(req, timeout=30))


def main() -> None:
    """Create (or reuse) the email channel and the 5xx alert policy."""
    chan_name = "paligemma-api alerts (email)"
    existing = _api("notificationChannels").get("notificationChannels", [])
    channel = next((c for c in existing if c.get("displayName") == chan_name), None)
    if channel is None:
        channel = _api(
            "notificationChannels",
            {
                "type": "email",
                "displayName": chan_name,
                "labels": {"email_address": ALERT_EMAIL},
            },
        )
        print(f"created channel {channel['name']} (verify via the email link)")
    else:
        print(f"channel exists: {channel['name']}")

    policy_name = f"{SERVICE} server errors (5xx)"
    policies = _api("alertPolicies").get("alertPolicies", [])
    if any(p.get("displayName") == policy_name for p in policies):
        print(f"policy exists: {policy_name}")
        return
    policy = _api(
        "alertPolicies",
        {
            "displayName": policy_name,
            "combiner": "OR",
            "conditions": [
                {
                    "displayName": "5xx responses > 0 over 5 min",
                    "conditionThreshold": {
                        "filter": (
                            'resource.type="cloud_run_revision" '
                            f'AND resource.labels.service_name="{SERVICE}" '
                            'AND metric.type="run.googleapis.com/request_count" '
                            'AND metric.label.response_code_class="5xx"'
                        ),
                        "aggregations": [
                            {"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_SUM"}
                        ],
                        "comparison": "COMPARISON_GT",
                        "thresholdValue": 0,
                        "duration": "0s",
                        "trigger": {"count": 1},
                    },
                }
            ],
            "notificationChannels": [channel["name"]],
            "alertStrategy": {"autoClose": "1800s"},
        },
    )
    print(f"created policy {policy['name']}")


if __name__ == "__main__":
    main()
