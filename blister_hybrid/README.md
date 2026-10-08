Hybrid AI Pipeline and Transfer Learning
(end-to-end industrial computer vision solution combining classical geometry with deep learning)

Pipeline Breakdown:
1. Automated Splicing: Extracts blister pack contours and slices images into individual cell crops.
2. Interactive Curation: GUI keyboard tool (f for filled, e for empty) to build ground truth training datasets.
3. Deep Learning Training: Utilizes PyTorch transfer learning (ResNet18), Adam optimizer, and cross-entropy loss to train a binary classifier on cell crops, saving weights as blister_model.pth.
4. Hybrid Inference: OpenCV maps the grid coordinates, feeds each cell crop through the AI model, and renders a live pass or defective inspection dashboard.
