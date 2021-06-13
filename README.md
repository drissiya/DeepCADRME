# DeepCADRME: A deep neural model for complex adverse drug reaction mentions extraction
DeepCADRME is a deep neural model for extracting complex adverse drug reaction (ADR) mentions (simple, nested, discontinuous and overlapping). It first transforms the ADR mentions extraction problem as an N-level tagging sequence. Then, it feeds the sequences to an N-level model based on contextual embeddings where the output of the pre-trained model of the current level is used to build a new deep contextualized representation for the next level. This allows the DeepCADRME system to transfer knowledge between levels.


