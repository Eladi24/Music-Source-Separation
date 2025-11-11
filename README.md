# Music Source Separation

Deep learning project for separating audio into isolated stems (vocals, drums, bass, other).

## Project Structure
- `notebooks/` - Jupyter notebooks for exploration and evaluation
- `src/` - Core Python modules
- `data/` - Audio datasets
- `models/` - Saved model checkpoints

## Setup
```bash
source ~/pytorch_env/bin/activate
pip install -r requirements.txt
```

## Usage
1. Explore data: `notebooks/01_data_exploration.ipynb`
2. Train model: `python src/train.py`
3. Inference: `python src/inference.py --input song.wav`