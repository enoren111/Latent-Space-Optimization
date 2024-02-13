# Latent-Space-Optimization
This project presents an innovative approach to enhance Fine-Grained Sketch-Based Image Retrieval (FG-SBIR) systems through latent space optimisation. Utilising the QMUL ShoeV2 dataset and leveraging a ResNet50 backbone, our method introduces three key improvements:
* warm-up training for dynamic learning rate adjustment
* stride modification for enriched feature maps
* the integration of Squeeze-and-Excitation (SE) attention mechanisms for enhanced model expressiveness

Our research demonstrates significant advancements in model accuracy and performance for FG-SBIR tasks, addressing the challenges of cross-modal retrieval and fine-grained feature discrimination. This repository includes the model implementation, training scripts, and evaluation protocols designed to foster further research and application in computer vision and sketch-based retrieval systems.

## Key Features:
* Warm-Up Training: Dynamically adjusts learning rates to improve model convergence and generalisation.
* Stride Modification: Enhances the granularity of feature maps, enabling the model to capture more detailed features.
* SE Attention Mechanisms: Boosts model expressiveness by adaptively re-weighting feature channels, focusing on the most relevant details for retrieval tasks.
