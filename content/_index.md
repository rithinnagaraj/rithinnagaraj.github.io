---
title: "About"
---

<section class="cv-section cv-bio">
<p>Hi, I'm Rithin. I'm a Computer Science undergraduate at PES University, Bangalore, and currently a research intern at <a href="https://lossfunk.com" target="_blank" rel="noopener">Lossfunk</a>, where I work on continual learning for sequential models under domain shift.</p>
<p>My research interests sit at the intersection of <strong>Reinforcement Learning</strong>, <strong>Continual Learning</strong> and <strong>Representation Learning</strong> — understanding how neural networks represent, retain, and adapt what they learn and developing methods to help networks retain previously learned knowledge while also being able to acquire new knowledge. I also write about these topics on my <a href="/posts/">blog</a>.</p>
</section>

<section class="cv-section">
<h2>Education</h2>
<div class="cv-item">
<div class="cv-item-head">
<h3>PES University, Bangalore</h3>
<span class="cv-date">2023 – 2027</span>
</div>
<p class="cv-sub">B.Tech in Computer Science and Engineering</p>
</div>
</section>

<section class="cv-section">
<h2>Research Experience</h2>
<div class="cv-item">
<div class="cv-item-head">
<h3>Research Intern · <a href="https://lossfunk.com" target="_blank" rel="noopener">Lossfunk</a></h3>
<span class="cv-date">Jun 2026 – Present</span>
</div>
<p class="cv-sub">Continual learning for sequential models under domain shift</p>
<ul>
<li>Designed a cosine-distance-based selective replay buffer for sequential models experiencing domain shift, identifying maximally diverse exemplars by comparing latent trajectory representations across domains.</li>
<li>Demonstrated that trajectory-level cosine similarity provides a more principled exemplar selection criterion than random replay, measurably reducing catastrophic forgetting on held-out prior-domain benchmarks.</li>
<li>Investigating the stability–plasticity trade-off in sequential architectures under non-stationary time series, targeting continual adaptation without episodic retraining from scratch.</li>
</ul>
<p class="cv-tech">PyTorch · RNN / LSTM / SSMs · Replay buffers · Time series analysis</p>
</div>
</section>

<section class="cv-section">
<h2>Publications</h2>
<div class="cv-item">
<div class="cv-item-head">
<h3>Mechanistic Interpretability in Large Language Models: SSMs vs. Transformers</h3>
<span class="cv-badge">IAIT 2026</span>
</div>
<ul>
<li>Mapped feature-level correspondences between Mamba-130m and Pythia-70m via SAE dictionary analysis over 10M text tokens, finding no evidence of architecturally exclusive "dark matter" features.</li>
<li>Identified a class of monosemantic landmark features unique to Mamba's recurrent hidden state, hypothesized to function as periodic state resets compensating for the absence of global positional attention.</li>
</ul>
<p class="cv-tech">PyTorch · Mamba · Pythia · Sparse Autoencoders · TransformerLens · MambaLens</p>
</div>
<div class="cv-item">
<div class="cv-item-head">
<h3>Pharmacokinetic SSMs for Haemodynamic Collapse Prediction</h3>
<span class="cv-badge cv-badge--accent">Best Paper Nominee · AIiH 2026</span>
</div>
<ul>
<li>Architected a Mamba-based selective state space model to forecast intraoperative hypotension 5–15 minutes in advance, achieving an AUROC of 0.7360 and a 2.73× AUPRC lift over random guessing.</li>
<li>Engineered explicit pharmacokinetic trajectories (Propofol and Remifentanil effect-site concentrations) as inputs, showing their inclusion prevents a 13.9% AUPRC drop versus haemodynamics-only baselines.</li>
<li>Quantified a critical "lead-gap contamination" selection bias in prior literature, empirically showing that failure to enforce strict temporal boundaries inflates apparent AUROC by 16.7%.</li>
</ul>
<p class="cv-tech">PyTorch · Mamba (SSMs) · Transformers · VitalDB</p>
</div>
</section>

<section class="cv-section">
<h2>Projects</h2>
<div class="cv-item">
<div class="cv-item-head">
<h3>Stable-Drift Re-implementation <a class="cv-link" href="https://github.com/rithinnagaraj/selective-buffer-cifar" target="_blank" rel="noopener">GitHub ↗</a></h3>
<span class="cv-date">Apr 2026</span>
</div>
<p class="cv-sub">Latent drift-guided replay (Theofilou et al., ICCV 2025)</p>
<ul>
<li>Re-implemented latent drift-guided replay: buffer exemplars are selected by cosine distance between a sample's internal representations before and after naive domain adaptation, prioritising the most representationally unstable samples.</li>
<li>Validated the core finding on CIFAR-10 under sequential domain shift — drift-based exemplar selection outperforms random replay in reducing catastrophic forgetting.</li>
</ul>
<p class="cv-tech">PyTorch · CNN · ViT · Replay buffers · CIFAR-10</p>
</div>
<div class="cv-item">
<div class="cv-item-head">
<h3>Probing Compositional Limits in Object-Centric Learning <a class="cv-link" href="https://github.com/rithinnagaraj/compositional-generalization-slots" target="_blank" rel="noopener">GitHub ↗</a></h3>
<span class="cv-date">Jan 2026</span>
</div>
<ul>
<li>Architected a decoupled vision–graph pipeline, integrating a frozen Slot Attention autoencoder with a custom pairwise relational network to extract and bind unsupervised object parts via connected components.</li>
<li>Achieved a 0.775 F1-score on a texture-flattened PartImageNet++ dataset by training the network to bridge over-segmented part artifacts using intra-domain geometric graph binding.</li>
<li>Demonstrated morphological limitations of unsupervised compositional architectures through a zero-shot domain transfer test (vehicles → quadrupeds), showing MLPs overfit to specific latent geometries.</li>
</ul>
<p class="cv-tech">PyTorch · Slot Attention · Graph Neural Networks · Scikit-learn · SciPy</p>
</div>
<div class="cv-item">
<div class="cv-item-head">
<h3>World Models Re-implementation <a class="cv-link" href="https://github.com/rithinnagaraj/world-models-implementation" target="_blank" rel="noopener">GitHub ↗</a></h3>
<span class="cv-date">Nov 2025</span>
</div>
<p class="cv-sub">Ha &amp; Schmidhuber, 2018</p>
<ul>
<li>Reproduced the full V+M+C pipeline from scratch on CarRacing-v0, replicating the three-phase training procedure: random rollouts → VAE → MDN-RNN → CMA-ES controller.</li>
<li>Trained the controller entirely within the MDN-RNN's dream space, confirming that imagined rollouts alone yield competitive policies.</li>
</ul>
<p class="cv-tech">PyTorch · VAE · MDN-RNN · CMA-ES · Gymnasium</p>
</div>
<div class="cv-item">
<div class="cv-item-head">
<h3>Mechanistic Analysis of Grokking in Modular Transformers <a class="cv-link" href="https://github.com/rithinnagaraj/neural-fourier-circuits" target="_blank" rel="noopener">GitHub ↗</a></h3>
<span class="cv-date">Nov 2025</span>
</div>
<ul>
<li>Investigated the grokking phase transition in 1-layer transformers on modular arithmetic, probing the circuit-level dynamics of delayed generalization.</li>
<li>Reverse-engineered internal representations using discrete Fourier transforms, demonstrating that the model spontaneously learns a trigonometric algorithm rather than rote memorization.</li>
<li>Visualized spectral specialization in embedding weights and oscillatory attention patterns to prove the emergence of neural Fourier circuits.</li>
</ul>
<p class="cv-tech">PyTorch · TransformerLens · Matplotlib</p>
</div>
</section>

<section class="cv-section">
<h2>Skills</h2>
<ul class="skills-list">
<li><strong>Deep Learning &amp; GenAI</strong> — PyTorch, Mamba (SSMs), TransformerLens, JAX, Flax, Optax, KANs</li>
<li><strong>Reinforcement Learning</strong> — World Models (VAE + MDN-RNN + CMA-ES), Gymnasium, foundational theory (Sutton &amp; Barto)</li>
<li><strong>Continual Learning</strong> — Replay buffer methods, cosine-distance exemplar selection, catastrophic forgetting mitigation, stability–plasticity trade-off</li>
<li><strong>Mechanistic Interpretability</strong> — Sparse autoencoders, circuit analysis, feature decomposition, TransformerLens, MambaLens</li>
<li><strong>Representation Learning</strong> — Slot Attention, object-centric learning, VAE, graph neural networks, disentangled representations</li>
<li><strong>Core &amp; Vision</strong> — Python, C/C++, Torchvision, OpenCV, NumPy</li>
<li><strong>Tools</strong> — Git, Jupyter, Matplotlib, Weights &amp; Biases</li>
</ul>
</section>
