# Postmortem: edge rate limiter HTTP 429 false positives

## Incident summary

On September 22, 2026, 18 percent of legitimate checkout traffic received `HTTP 429` responses for 11 minutes following an edge CDN configuration push.

## Root cause

An upstream CDN rule stripped the trusted `X_Real_Client_IP` header and left only the regional PoP proxy address in `X_Forwarded_For`. The token bucket rate limiter keyed buckets on the fallback proxy IP, collapsing thousands of distinct shoppers into 6 PoP buckets that exhausted the 300 requests per minute per client ceiling within seconds.

## Corrective actions

Edge Envoy filters now validate proxy hop count against signed CDN mTLS identities and fail open to merchant API key bucketing whenever client IP extraction falls back to an internal PoP subnet.
