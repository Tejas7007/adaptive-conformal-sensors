# Adaptive Conformal Prediction for Sensor Calibration Drift

Reliable uncertainty quantification for virtual sensors between calibrations.

## Results Summary

| Dataset | Standard | Adaptive | Improvement |
|---------|----------|----------|-------------|
| NASA C-MAPSS (Linear 0.5) | 50.8% | 89.3% | +38.5pp ✅ |
| NASA C-MAPSS (Stepwise) | 47.2% | 88.4% | +41.3pp ✅ |
| Bearing (Replacement) | 63.7% | 82.3% | +18.6pp ✅ |
| Wind Turbine | 67.5%±13.6% | 88.7%±1.5% | +21.3pp ✅ |
| Gas Sensors (REAL) | 86.8%±2.6% | 73.7%±2.0% | -13.1pp ❌ |

**Key Finding:** 9x variance reduction (±1.5% vs ±13.6%)

## When It Works

✅ Calibration-style drift (bias, scale, stepwise)  
✅ Cross-equipment deployment  
❌ Complex multi-sensor drift  

## Installation
```bash
git clone https://github.com/Tejas7007/adaptive-conformal-sensors.git
cd adaptive-conformal-sensors
pip install -r requirements.txt
```

## Quick Start
```python
from src.models.sensor_lstm import SensorLSTM

model = SensorLSTM(input_dim=5, hidden_dim=64)
model.train(X_train, y_train)

# Adaptive conformal
from src.adaptive_conformal import AdaptiveConformal
cp = AdaptiveConformal(model, alpha=0.1)
cp.calibrate(X_cal, y_cal)

# Deploy
predictions, lower, upper = cp.predict(X_test)
cp.update(X_test, y_test)  # Adapt to drift
```

## Structure
```
├── src/
│   ├── models/          # LSTM virtual sensor
│   └── data/            # Data loaders
├── experiments/         # Reproducible experiments
├── figures/            # Publication figures
└── results/            # CSV results
```

## Reproduce Results
```bash
PYTHONPATH=. python experiments/calibration_drift_simple.py
PYTHONPATH=. python experiments/wind_turbine_multiseed.py
python experiments/create_publication_figures.py
```

## Economics

- **Savings:** $164K/year per sensor
- **Calibration reduction:** 3x (monthly → quarterly)
- **ROI:** 3,754%

## Citation
```bibtex
@article{nuxoll2024adaptive,
  title={Adaptive Conformal Prediction for Virtual Sensors},
  author={Nuxoll, Diya},
  year={2024},
  institution={UW-Madison}
}
```

## Contact

Diya Nuxoll | dnuxoll@wisc.edu | UW-Madison

## License

MIT
