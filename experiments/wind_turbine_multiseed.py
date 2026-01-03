"""
Wind turbine with multiple seeds for error bars
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.models.sensor_lstm import SensorLSTM

def load_wind_data(seed):
    """Generate wind turbine data with specific seed"""
    np.random.seed(seed)
    n_samples = 5000
    time = np.linspace(0, 365, n_samples)
    
    wind_speed = 8 + 3*np.sin(2*np.pi*time/365) + 2*np.sin(2*np.pi*time) + np.random.randn(n_samples)*0.5
    wind_speed = np.maximum(wind_speed, 0)
    rotor_speed = 10 + 0.8*wind_speed + np.random.randn(n_samples)*0.3
    power = 0.5 * rotor_speed**2 + np.random.randn(n_samples)*10
    temp = 15 + 10*np.sin(2*np.pi*time/365) + np.random.randn(n_samples)*2
    vibration = 0.1 + (time/365)*0.05 + np.random.randn(n_samples)*0.02
    
    # REAL DRIFT
    temp_drift = (time/365) * 0.5
    temp = temp + temp_drift
    vibration_gain_drift = 1 + (time/365) * 0.15
    vibration = vibration * vibration_gain_drift
    
    df = pd.DataFrame({
        'wind_speed': wind_speed,
        'rotor_speed': rotor_speed,
        'power': power,
        'temperature': temp,
        'vibration': vibration
    })
    
    return df

def test_wind_turbine_seed(seed):
    """Run wind turbine experiment with specific seed"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    df = load_wind_data(seed)
    
    sequence_length = 24
    X_seq, y_seq = [], []
    
    feature_cols = ['wind_speed', 'rotor_speed', 'temperature', 'vibration']
    target_col = 'power'
    
    for i in range(len(df) - sequence_length):
        X_seq.append(df.iloc[i:i+sequence_length][feature_cols].values)
        y_seq.append(df.iloc[i+sequence_length][target_col])
    
    X = np.array(X_seq)
    y = np.array(y_seq)
    
    X = (X - X.mean(axis=(0,1))) / (X.std(axis=(0,1)) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    
    n = len(X)
    n_early = int(0.4 * n)
    n_late_start = int(0.6 * n)
    
    X_early = X[:n_early]
    y_early = y[:n_early]
    X_late = X[n_late_start:]
    y_late = y[n_late_start:]
    
    device = torch.device('cpu')
    X_early_t = torch.FloatTensor(X_early).to(device)
    y_early_t = torch.FloatTensor(y_early).to(device)
    X_late_t = torch.FloatTensor(X_late).to(device)
    y_late_t = torch.FloatTensor(y_late).to(device)
    
    input_dim = X.shape[2]
    model = SensorLSTM(input_dim, 32, 2, 0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        pred = model(X_early_t).squeeze()
        loss = nn.MSELoss()(pred, y_early_t)
        loss.backward()
        optimizer.step()
    
    n_cal = len(X_early) // 2
    X_cal = X_early_t[:n_cal]
    y_cal = y_early_t[:n_cal]
    
    model.eval()
    with torch.no_grad():
        y_pred_cal = model(X_cal).squeeze()
    
    scores = torch.abs(y_cal - y_pred_cal)
    alpha = 0.1
    q_level = np.ceil((len(scores)+1)*(1-alpha))/len(scores)
    quantile = torch.quantile(scores, q_level)
    
    with torch.no_grad():
        y_pred_std = model(X_late_t).squeeze()
    
    covered_std = (y_late_t >= y_pred_std - quantile) & (y_late_t <= y_pred_std + quantile)
    coverage_std = covered_std.float().mean().item()
    
    model_adp = SensorLSTM(input_dim, 32, 2, 0.2).to(device)
    model_adp.load_state_dict(model.state_dict())
    optimizer_adp = torch.optim.Adam(model_adp.parameters(), lr=1e-4)
    
    n_adapt = min(200, len(X_late) // 3)
    X_adapt = X_late_t[:n_adapt]
    y_adapt = y_late_t[:n_adapt]
    
    for _ in range(15):
        optimizer_adp.zero_grad()
        pred = model_adp(X_adapt).squeeze()
        loss = nn.MSELoss()(pred, y_adapt)
        loss.backward()
        optimizer_adp.step()
    
    model_adp.eval()
    with torch.no_grad():
        y_pred_adapt_cal = model_adp(X_adapt).squeeze()
    
    scores_adapt = torch.abs(y_adapt - y_pred_adapt_cal)
    quantile_adapt = torch.quantile(scores_adapt, q_level)
    
    X_test = X_late_t[n_adapt:]
    y_test = y_late_t[n_adapt:]
    
    with torch.no_grad():
        y_pred_adp = model_adp(X_test).squeeze()
    
    covered_adp = (y_test >= y_pred_adp - quantile_adapt) & (y_test <= y_pred_adp + quantile_adapt)
    coverage_adp = covered_adp.float().mean().item()
    
    return {
        'seed': seed,
        'coverage_std': coverage_std,
        'coverage_adp': coverage_adp,
        'improvement': coverage_adp - coverage_std
    }

def main():
    print("="*70)
    print("WIND TURBINE - Statistical Validation (5 seeds)")
    print("="*70)
    
    results = []
    
    for seed in range(5):
        print(f"\nSeed {seed+1}/5...")
        result = test_wind_turbine_seed(seed)
        results.append(result)
        print(f"  Std: {result['coverage_std']:.1%}, Adp: {result['coverage_adp']:.1%}, Δ: {result['improvement']*100:+.1f}pp")
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("RESULTS WITH ERROR BARS:")
    print(f"{'='*70}")
    print(f"Standard: {df['coverage_std'].mean():.1%} ± {df['coverage_std'].std():.1%}")
    print(f"Adaptive: {df['coverage_adp'].mean():.1%} ± {df['coverage_adp'].std():.1%}")
    print(f"Improvement: {df['improvement'].mean()*100:+.1f} ± {df['improvement'].std()*100:.1f}pp")
    print(f"{'='*70}")
    
    df.to_csv('wind_turbine_multiseed.csv', index=False)
    print("\n✓ Saved: wind_turbine_multiseed.csv")

if __name__ == '__main__':
    main()
