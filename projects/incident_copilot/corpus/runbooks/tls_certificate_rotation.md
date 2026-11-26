# Edge mTLS leaf certificate rotation

## Symptoms and paging alerts

The alert `CertExpiryUnder72Hours` triggers when an internal gateway or partner mTLS leaf certificate has fewer than 72 hours of validity left, or when ACME renewal jobs stay in `Pending` for over 30 minutes.

## Diagnosis

Inspect `CertificateRequest` status events in the edge control plane. Stale DNS TXT records or rate limited ACME orders are the primary causes of stalled automatic renewals.

## Immediate mitigation

Issue a fresh leaf certificate from the internal Vault PKI role `atlas_edge_leaf` with a 30 day validity window, write the bundle to Kubernetes secret `edge_mtls_bundle`, and trigger Envoy SDS hot reload without dropping active HTTP2 streams. Verify the presented serial number across all ingress PoPs before closing the incident.
