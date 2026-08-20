# Mamba-CP: Continuous-Time Sequence Modeling and Empirical Conformal Prediction for Exoplanet Transit Detection

**Author:** Ahmed Mohamed Elmohamedy Khalfalla

## Abstract

Traditional exoplanet transit detection relies on computationally expensive phase-folding algorithms (e.g., Box-fitting Least Squares) that struggle with continuous-time thermodynamic stellar variability. This project introduces a self-supervised sequence modeling pipeline paired with Empirical Conformal Prediction to detect exoplanet transits dynamically. By applying a digital moving-median filter to isolate high-frequency residuals, a Gated Recurrent Unit (GRU) backbone was trained to forecast the nominal stellar baseline. An empirical conformal bound was mathematically derived to target the 3.2nd percentile of prediction errors, creating a highly sensitive anomaly envelope. Tested on continuous long-cadence telemetry from the Kepler Space Telescope, the model successfully isolated the transit signatures of Kepler-10b, algorithmically extracting an orbital period of 0.7928 days (a ~5% error margin from the true 0.837-day period) and deriving a planetary radius of 1.41 Earth Radii.

**Index Terms:** Digital Signal Processing, Empirical Conformal Prediction, Exoplanet Detection, Gated Recurrent Units, Time-Series Forecasting.

---

## 1. Introduction: The Exoplanet Detection Bottleneck

Searching for exoplanets using the transit method is fundamentally a massive, continuous-time anomaly detection problem. Space telescopes like Kepler and TESS stare at distant stars, recording their brightness (flux) over time. When a planet crosses in front of its host star, it creates a minuscule dip in the light curve—often obscuring less than 0.01% of the total stellar flux.

### The Traditional Approach and its Limits

For the past decade, the astrophysics community has relied on periodograms and phase-folding algorithms, most notably the **Box-fitting Least Squares (BLS)** algorithm. While highly accurate, BLS is computationally expensive. It requires searching through a massive grid of potential orbital periods, durations, and phases, resulting in an $O(N^2)$ computational bottleneck. Furthermore, standard algorithms struggle to dynamically separate the high-frequency planetary transits from the low-frequency thermodynamic "breathing" (rotation and starspot activity) of the host star.

### The Deep Learning Challenge

Applying modern deep learning to this problem introduces a new set of challenges:

1. **Signal Explosion:** Continuous-time sequence models (like true State-Space Models) without proper internal gating or hardware-level parallel associative scans can suffer from recursive signal explosion when processing sequences of 20,000+ time steps.
2. **Conformal Bound Collapse:** Because transit dips represent such a tiny fraction of the dataset (typically < 2%), networks tasked with predicting their own uncertainty boundaries often experience "bound collapse." To avoid penalization, the loss function artificially widens the envelope until it swallows the anomalies entirely.

### The Mamba-CP Solution

**Mamba-CP** was engineered to solve these exact bottlenecks by bridging classical Digital Signal Processing (DSP) with modern Deep Learning and rigorous statistical math.

Instead of relying on a neural network to do everything, this pipeline delegates specific tasks to the optimal mathematical layer:

* **DSP Pre-Conditioning:** A dynamic digital moving-median filter automatically flattens the low-frequency stellar baseline, allowing the model to focus strictly on high-frequency residuals.
* **Sequence Modeling:** A stable Gated Recurrent Unit (GRU) acts as the sequence mixer, utilizing its internal sigmoid gates as robust Automatic Gain Control to prevent signal explosion on local CPU hardware.
* **Empirical Conformal Prediction:** Instead of forcing the neural network to predict anomaly boundaries internally, the boundary is mathematically calculated post-prediction. By isolating the 3.2nd percentile of the model's residual errors, the pipeline guarantees a mathematically rigid envelope that captures "grazing" transits without collapsing.

---

## 2. Methodology & Limitations

### Core Architecture

* **Digital Pre-Conditioning:** A dynamic moving-median filter (equivalent to a discrete low-pass filter) was applied to subtract low-frequency stellar thermodynamic rotation, feeding only high-frequency residuals to the network.
* **Sequence Modeling:** A next-step forecasting architecture optimizing Mean Squared Error (MSE) to learn the flattened stellar baseline.
* **Empirical Conformal Prediction:** Post-prediction statistical bounding at the 3.2% threshold to guarantee dynamic envelope stability.
* **Temporal Clustering:** Time-delta grouping of anomalous points to calculate multi-day orbital periods from continuous-time sensor data.

### Hardware & Computational Limitations

The original architecture was designed utilizing a true Selective State Space Model (SSM / Mamba). However, simulating continuous-time differential equations sequentially on a local CPU leads to severe processing bottlenecks, particularly when scaling to the 20,000+ time-step sequences required by high-cadence surveys. True SSM layers rely on hardware-aware, parallel associative scans that require custom CUDA kernel compilation on dedicated GPUs. To ensure local execution stability and prevent signal explosion, the pipeline successfully fell back to a GRU, utilizing its internal sigmoid gates for robust Automatic Gain Control at the cost of parallel processing speed.

### Sensor Telemetry Limitations (TESS vs. Kepler)

While the pipeline effectively decoupled the mathematical cadence parameters from the codebase, high-resolution tests on the Transiting Exoplanet Survey Satellite (TESS) revealed the limits of simple digital pre-conditioning. Unlike Kepler's stable Earth-trailing orbit, TESS occupies a highly elliptical Earth orbit, requiring frequent momentum dumps (thruster firings) to stabilize its reaction wheels. These mechanical jitters introduce massive, high-frequency anomalies that mimic exoplanet transits. Deploying this pipeline on raw TESS telemetry requires the integration of advanced instrumental detrending prior to neural ingestion to differentiate between actual planetary silhouettes and spacecraft mechanical noise.

---

## 3. Installation

This project is built using Python 3.10+ and relies on PyTorch for sequence modeling and Lightkurve for fetching NASA telemetry. To ensure complete reproducibility, it is highly recommended to run this pipeline inside an isolated virtual environment.

**1. Clone the repository:**

```bash
git clone https://github.com/yourusername/mamba-cp-exoplanet.git
cd mamba-cp-exoplanet

```

**2. Create and activate a virtual environment:**

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

```

**3. Install the required dependencies:**

```bash
pip install -r requirements.txt

```

---

## 4. Usage & Execution

The pipeline is modularly designed. You can fetch raw telemetry from NASA's MAST archive and run the conformal prediction pipeline with just two commands.

### Fetching TESS Data

To demonstrate the pipeline's dynamic digital signal processing, you can fetch high-resolution (2-minute cadence) data from the Transiting Exoplanet Survey Satellite (TESS).

The included fetch script automatically targets Pi Mensae, cleans the raw SPOC-pipeline telemetry, normalizes the flux, and saves it as a Parquet file for high-speed ingestion:

```bash
python scripts/fetch_tess.py

```

*(Note: If you wish to test Kepler data, ensure your long-cadence `.parquet` files are placed directly in the `/data` directory).*

### Running the Detection Pipeline

The core transit detection logic is housed in `detect_transits.py`.

Before running, ensure that the `TARGET_CADENCE` variable at the top of the `main()` function matches your dataset's sampling rate (e.g., `2.0` for TESS, `29.4` for Kepler). The pipeline uses this parameter to automatically calculate the digital filter window and the astrophysical clustering thresholds.

```bash
python scripts/detect_transits.py

```

### Expected Output

Upon execution, the PyTorch Lightning trainer will optimize the GRU backbone to learn the flattened stellar baseline. Once trained, the empirical conformal bound is calculated and transits are flagged.

The terminal will output the astrophysical metrics:

```text
Total anomaly points detected: 25
Clustered into 16 distinct planetary transit events.
--> Estimated Orbital Period: 0.7928 Earth Days

```

Additionally, the script generates a high-resolution visualization (`detected_transits_plot.png`), showcasing the continuous-time nominal flux, the predicted baseline, the 3.2% conformal lower bound, and the successfully flagged transit events.

---

## 5. Conclusion and Future Work

The **Mamba-CP** pipeline successfully demonstrates that continuous-time sequence modeling, when rigorously constrained by empirical statistical math, can autonomously detect exoplanet transits without the heavy computational overhead of traditional phase-folding algorithms.

Tested on Kepler Space Telescope telemetry, the pipeline extracted the orbital period of Kepler-10b to within a ~5% error margin (0.7928 days calculated vs. 0.837 days confirmed) and algorithmically derived a planetary radius of 1.41 Earth Radii (vs. 1.47 confirmed).

### Future Work & Architectural Scaling

To transition this pipeline from a local, single-target sequence model to a massive, survey-scale astrophysical tool, future development will focus on three distinct engineering pillars:

**1. Hardware-Accelerated State Space Models (The Mamba Upgrade)**
While the GRU provided stability, its recurrent nature forces sequential processing, creating a severe CPU bottleneck when scaling to 20,000+ time-step sequences. Future iterations will replace the GRU with a true Continuous-Time Selective State Space Model (SSM). This requires migrating execution to dedicated GPUs to leverage custom CUDA kernels for hardware-aware, parallel associative scans, allowing the model to process months of continuous telemetry simultaneously without recursive signal explosion.

**2. Advanced Instrumental Detrending (TESS Integration)**
To scale this pipeline to TESS, the architecture must be prepended with an advanced instrumental detrending module utilizing Co-Trending Basis Vectors (CBVs) or Pixel Level Decorrelation (PLD) to systematically subtract spacecraft mechanical noise prior to neural ingestion.

**3. Multi-Planet System Clustering**
The current temporal clustering algorithm is optimized for single-planet isolation. Future versions will incorporate multi-dimensional clustering logic (incorporating both transit duration and transit depth) to disentangle overlapping transits in complex, multi-planet systems like TRAPPIST-1, applying independent conformal thresholds for each planetary signature.

---
