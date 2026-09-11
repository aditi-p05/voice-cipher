# Architecture boundary

```text
Voice call / Chatbot / Complaint portal
                |
                v
      Layer 0: InputEnvelope
                |
                v
      Layer 1: EvidenceBundle
                |
                v
      Layer 2: SVIResult + risk tier
                |
                v
      Layer 4: ServiceRecommendation / gated Support
                |
                v
      Layer 5: Dashboard and OperatorAction
```

Layer 0 is currently the only implemented layer. Its code remains in `backend/` because that is its existing, working structure. Future teams may choose their own internal file layouts, provided they preserve the shared contracts in the repository root.
