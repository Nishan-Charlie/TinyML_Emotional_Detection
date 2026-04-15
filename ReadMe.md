<!DOCTYPE html>
<html lang="en">

<body>

<header>
<img src="docs/images/images.png" width="85">

<h3>Projects - Department of Computing and Information System</h3>
<h4>University of Sabaragamuwa, Sri Lanka</h4>
</header>

<hr>

<h1>Human Emotion Detection with LightCBAMNet: A Tiny Deep Learning Model for Edge Devices</h1>

<p>
A lightweight facial emotion recognition framework designed for resource-constrained devices,
combining efficient preprocessing, knowledge distillation, and compact model design for
accurate edge deployment.
</p>

<hr>

<h3>Team</h3>
<ul>
<li>T. Sajeeth - tsajeeth@std.appsc.sab.ac.lk</li>
</ul>

<h3>Supervisors</h3>
<ul>
<li>Lec. Nishankar S - Nishankar@eng.pdn.ac.lk</li>
<li>Prof. Vigneshwaran P - Vigneshwaran@eng.pdn.ac.lk</li>
</ul>

<hr>

<h3>Table of Content</h3>
<ol>
<li>Abstract & Overview</li>
<li>Key Innovations</li>
<li>Data Preparation & Augmentation</li>
<li>Methodology</li>
<li>Architecture Breakdown</li>
<li>Training Progress</li>
<li>Results & Visual Analysis</li>
<li>Comparative Analysis</li>
<li>Deployment & Quantization</li>
<li>Setup & Implementation</li>
<li>Links</li>
</ol>

<hr>

<h2>1. Abstract & Overview</h2>

<p>
Facial Emotion Recognition (FER) on edge devices requires a balance between accuracy and computational efficiency.
Traditional deep learning models rely on high-performance GPUs, making them unsuitable for real-time, low-power
IoT environments. This study proposes an efficient FER framework using the FER-2013 dataset, designed specifically
for edge deployment.
</p>

<p>
The approach includes a preprocessing pipeline with CLAHE, unsharp masking, facial alignment, and masking to
enhance feature quality. A dual-teacher knowledge distillation strategy is applied, transferring knowledge from
ResNet50 and VGG19 models to a lightweight MobileNetV4-based student model.
</p>

<p>
The proposed model achieves 72.0% accuracy, outperforming both teacher models and ensemble methods, while using
only 3.8M parameters and 4–8 ms inference time per image. This enables real-time, privacy-preserving emotion
recognition on edge devices.
</p>

<hr>

<h2>2. Key Innovations</h2>

<h3>Phish Activation</h3>
<p>x · tanh(GELU(x))</p>

<h3>DCWP Module</h3>
<p>Depthwise Convolution with Phish activation reduces FLOPs while preserving meaningful spatial and channel-level features.</p>

<h3>CBAM Attention</h3>
<p>The Convolutional Block Attention Module helps the network focus on the most important facial regions such as the eyes, eyebrows, nose, and mouth.</p>

<h3>Edge Efficiency</h3>
<p>The model is designed for small hardware environments where fast inference, lower latency, and reduced resource usage are essential.</p>

<hr>

<h2>3. Data Preparation & Augmentation</h2>

<h3>3.1 Original Facial Emotion Samples</h3>
<p>
The original FER images contain faces with different poses, expressions, brightness levels,
and background conditions. These raw images show the natural variability found in real-world
emotion recognition datasets.
</p>

<img src="docs/images/original.jpeg" width="500">

<p>Figure 1: Original facial emotion samples from the dataset before preprocessing.</p>

<h3>3.2 Preprocessing Stage 1 - Contrast Enhancement</h3>

<p>
In the first stage, image quality is improved using methods such as CLAHE and unsharp masking.
This helps make important facial details more visible by improving local contrast and edge clarity.
</p>

<img src="docs/images/pre_process_1.png" width="500">

<p>Figure 2: Enhanced images after applying contrast improvement and sharpening operations.</p>

<h3>3.3 Preprocessing Stage 2 - Facial Alignment</h3>

<p>
In the second stage, the face is aligned so that major landmarks appear in a more consistent position.
This reduces unnecessary variation caused by head tilt or image orientation.
</p>

<img src="docs/images/pre_process_2.png" width="500">

<p>Figure 3: Facial alignment step used to normalize the facial region.</p>

<h3>3.4 Preprocessing Stage 3 - Final Normalized Output</h3>

<p>
After enhancement and alignment, the final processed images become cleaner and more consistent.
This allows the model to learn emotion-related features more effectively.
</p>

<img src="docs/images/Pre_process_3.png" width="500">

<p>Figure 4: Final preprocessed samples used for training after normalization and refinement.</p>

<h3>3.5 Facial Masking</h3>

<p>
Masking is applied to focus the model more strongly on the face region and reduce background influence.
This helps improve feature quality by removing less useful surrounding information.
</p>

<img src="docs/images/masked.png" width="500">

<p>Figure 5: Masked facial samples highlighting the region of interest used for training.</p>

<hr>

<h2>4. Methodology</h2>

<p>
The overall workflow begins with raw facial images. These images are first preprocessed through
enhancement, facial alignment, and masking. Then, knowledge distillation is performed using
ResNet50 and VGG-16 as teacher models. Their soft-label knowledge is transferred to the
lightweight student model, MobileNetV4 / LightCBAMNet.
</p>

<p>
During training, the student model learns from both soft labels generated by the teachers and
hard labels from the actual dataset. This helps the compact model achieve better generalization
while remaining computationally efficient.
</p>

<img src="docs/images/methodology-diagram.png" width="600">

<p>Figure 6: Research methodology showing preprocessing, distilled knowledge transfer, student learning, and prediction workflow.</p>

<hr>

<h2>5. Architecture Breakdown</h2>

<ul>
<li>Input Stage: Receives preprocessed facial images</li>
<li>Feature Extraction: Lightweight convolution blocks capture local facial patterns</li>
<li>Attention Mechanism: CBAM emphasizes important regions</li>
<li>Compact Representation: Reduces model size</li>
<li>Classifier Head: Outputs emotion prediction</li>
</ul>

<hr>

<h2>6. Training Progress</h2>

<p>
The training graph shows that the model learns quickly in the early epochs and then gradually stabilizes.
Training accuracy remains high, while validation accuracy reaches approximately 72.68%.
</p>

<img src="docs/images/Training_graph.png" width="500">

<p>Figure 7: Training graph.</p>

<hr>

<h2>7. Results & Visual Analysis</h2>

<h3>7.1 Confusion Matrix</h3>

<img src="docs/images/confusion_matrix.png" width="500">

<p>Figure 8: Confusion matrix.</p>

<h3>7.2 Feature Visualization</h3>

<img src="docs/images/feature.jpeg" width="500">

<p>Figure 9: Feature visualization.</p>

<hr>

<h2>8. Comparative Analysis</h2>

<p>Performance comparison across models.</p>

<hr>

<h2>9. Deployment & Quantization</h2>

<img src="docs/images/deploye.png" width="500">

<p>Figure 10: Deployment pipeline.</p>

<hr>

<h2>10. Setup & Implementation</h2>

<pre>
pip install torch torchvision scikit-learn matplotlib numpy tqdm
</pre>

<pre>
python train.py --data_dir images --epochs 100 --batch_size 32 --lr 0.001
</pre>

<hr>

<h2>11. Links</h2>

<ul>
<li>https://github.com/Nishan-Charlie/TinyML_Emotional_Detection.git</li>
</ul>

<hr>

<p>
© 2026 Department of Computing and Information System  
University of Sabaragamuwa, Sri Lanka
</p>

</body>
</html>