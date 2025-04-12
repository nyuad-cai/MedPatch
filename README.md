## Code for MedPatch: Confidence-Guided Multi-Stage Fusion for Multimodal Clinical Data


Background
============
Multi-modal fusion approaches have been widely adopted to combine complementary information from diverse data types. In fields such as audio-visual processing, modalities are naturally paired and captured simultaneously. In contrast, healthcare data is often collected asynchronously and exhibits substantial heterogeneity. Patient records in electronic health records (EHRs), chest X-ray (CXR) images, radiology reports (RR), and discharge notes (DN) are typically acquired at different times and under different conditions. This inherent asynchrony makes it impractical to require that all modalities be present at every stage of training or inference.

To address these challenges, we introduce MedPatch, a conceptually simple yet effective multi-stage fusion network designed to accommodate both uni-modal and multi-modal inputs. MedPatch incorporates a token-level confidence mechanism to dynamically group and fuse modality-specific representations. By partitioning tokens into high- and low-confidence clusters, and by explicitly incorporating a missingness module, MedPatch not only leverages the strengths of each individual modality but also remains robust when parts of the data are missing. This approach leads to significant performance improvements on clinical prediction tasks such as in-hospital mortality and phenotype classification.


