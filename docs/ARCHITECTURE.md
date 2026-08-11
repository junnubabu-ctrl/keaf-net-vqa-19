# Architecture

KEAF-Net consumes region features, a tokenized question, and candidate knowledge facts.

1. **Encoding:** modality-specific projections map region and fact features into a shared space; question tokens are mask-pooled.
2. **AKF:** each fact is scored from the fact, question, interaction, and absolute difference. Masked top-*k* facts are retained.
3. **HGAF:** typed visual, question, and fact nodes exchange question-gated attention messages with residual normalization.
4. **MHSR:** a recurrent controller attends over the fused graph for a configured number of hops (three by default).
5. **Prediction:** the final reasoning state is normalized and mapped to the answer vocabulary.

Forward outputs include selected fact indices, filter weights, graph attention, and hop attention so evaluation can audit the evidence path. Attention values are diagnostic signals, not automatically faithful causal explanations.
