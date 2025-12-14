# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Nagy Boldizsár
- **Aiming for +1 Mark**: No

### Solution Description

This project implements a classifier to categorize financial chart patterns, focusing on "Bull flag" and "Bear flag" formations.

**Problem Statement**:
Financial analysts rely on visual chart patterns to predict market movements.

**Model Architecture**:
The solution utilizes a **Multi-Layer Perceptron (MLP)** classifier implemented with scikit-learn.
- **Input**: Flattened time-series segments (resampled to 100 steps $\times$ 4 features: Open, High, Low, Close).
- **Hidden Layers**: Two hidden layers with 128 and 64 neurons, respectively.
- **Activation**: ReLU.
- **Output**: Classification of specific chart patterns.

**Methodology**:
1.  **Data Processing**: The raw data (JSON annotations and CSV price history) is processed to extract specific time segments. A "fuzzy matching" algorithm links annotations to the correct CSV files.
2.  **Normalization**: Segments are resampled to a fixed length (100 steps) using linear interpolation and normalized relative to the starting price to ensure the model learns the *shape* of the pattern rather than absolute price levels.
3.  **Training**: The model is trained on an 80/20 train/test split.
4.  **Evaluation**: The model is evaluated on unseen test data, producing detailed classification reports and a "Binary Confusion Matrix" to assess its ability to distinguish Bullish vs. Bearish trends.

**Results**:
The data is inconsistent and the flags are not that reliable, the reading was pretty hard. The classifier distuingishes Bullish and Bearish quite well, but the accuracy is much lower on the subcategories.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t bullflag_classifier .
