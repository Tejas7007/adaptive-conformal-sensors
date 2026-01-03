import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.data.real_cmapss_loader import prepare_real_cmapss_data
from src.models.sensor_lstm import SensorLSTM

def apply_drift(y, drift_type, magnitude):
    n = len(y)
    time = torch.linspace(0, 1, n)
    
    if drift_type == 'linear_bias':
        drift = magnitude * time
    elif drift_type == 'scale_drift':
        drift = y * (magnitude * time)
    elif drift_type == 'exponential':
        drift = magnitude * (torch.exp(2*time) - 1) / (np.e**2 - 1)
    elif drift_type == 'stepwise':
        drift = torch.zeros(n)
        drift[time >= 0.3] = magnitude * 0.3
        drift[time >= 0.7] = magnitude * 0.7
    else:
        drift = torch.zeros(n)
    
    return y + drift

def test_one(drift_type, magnitude):
    train_loader, val_loader, test_loader, _ = prepare_real_cmapss_data()
    device = torch.device('cpu')
    input_dim = next(iter(train_loader))[0].shape[2]
    
    # Load model
    model = SensorLSTM(input_dim, 64, 2, 0.2).to(device)
    model.load_state_dict(torch.load('best_real_sensor_model.pt'))
    
    # Get test data
    X_test_all, y_test_all = [], []
    for X, y in test_loader:
        X_test_all.append(X)
        y_test_all.append(y)
    
    X_test = torch.cat(X_test_all, dim=0).to(device)
    y_test = torch.cat(y_test_all, dim=0).to(device)
    
    # Apply drift
    y_test_drifted = apply_drift(y_test, drift_type, magnitude)
    
    # Calibrate
    n_cal = int(0.15 * len(X_test))
    X_cal = X_test[:n_cal]
    y_cal = y_test_drifted[:n_cal]
    
    model.eval()
    with torch.no_grad():
        y_pred_cal = model(X_cal).squeeze()
    
    scores = torch.abs(y_cal - y_pred_cal)
    alpha = 0.1
    q_level = np.ceil((len(scores)+1)*(1-alpha))/len(scores)
    quantile = torch.quantile(scores, q_level)
    
    X_eval = X_test[n_cal:]
    y_eval = y_test_drifted[n_cal:]
    
    # Standard conformal
    with torch.no_grad():
        y_pred = model(X_eval).squeeze()
    
    covered_std = (y_eval >= y_pred - quantile) & (y_eval <= y_pred + quantile)
    coverage_std = covered_std.float().mean().item()
    
    # Adaptive conformal (reload model fresh)
    model_adp = SensorLSTM(input_dim, 64, 2, 0.2).to(device)
    model_adp.load_state_dict(torch.load('best_real_sensor_model.pt'))
    optimizer = torch.optim.Adam(model_adp.parameters(), lr=1e-4)
    
    batch_size = 50
    all_preds, all_lowers, all_uppers = [], [], []
    residual_buffer = []
    X_buffer, y_buffer = [], []
    current_quantile = quantile
    
    for i in range(0, len(X_eval), batch_size):
        X_batch = X_eval[i:i+batch_size]
        y_batch = y_eval[i:i+batch_size]
        
        model_adp.eval()
        with torch.no_grad():
            y_pred_batch = model_adp(X_batch).squeeze()
        
        all_preds.append(y_pred_batch)
        all_lowers.append(y_pred_batch - current_quantile)
        all_uppers.append(y_pred_batch + current_quantile)
        
        X_buffer.append(X_batch)
        y_buffer.append(y_batch)
        if len(X_buffer) > 2:
            X_buffer.pop(0)
            y_buffer.pop(0)
        
        residuals = torch.abs(y_batch - y_pred_batch)
        residual_buffer.extend(residuals.cpu().tolist())
        if len(residual_buffer) > 100:
            residual_buffer = residual_buffer[-100:]
        
        if i % 100 == 0 and i > 0 and len(X_buffer) >= 2:
            X_recent = torch.cat(X_buffer, dim=0)
            y_recent = torch.cat(y_buffer, dim=0)
            
            model_adp.train()
            for _ in range(3):
                optimizer.zero_grad()
                pred = model_adp(X_recent).squeeze()
                loss = nn.MSELoss()(pred, y_recent)
                loss.backward()
                optimizer.step()
            
            model_adp.eval()
            with torch.no_grad():
                y_pred_new = model_adp(X_recent).squeeze()
            new_residuals = torch.abs(y_recent - y_pred_new)
            residual_buffer = new_residuals.cpu().tolist()
            
            if len(residual_buffer) >= 50:
                recent_residuals = torch.tensor(residual_buffer).to(device)
                new_quantile = torch.quantile(recent_residuals, q_level)
                current_quantile = 0.5 * current_quantile + 0.5 * new_quantile
    
    all_preds = torch.cat(all_preds)
    all_lowers = torch.cat(all_lowers)
    all_uppers = torch.cat(all_uppers)
    
    covered_adp = (y_eval >= all_lowers) & (y_eval <= all_uppers)
    coverage_adp = covered_adp.float().mean().item()
    
    return coverage_std, coverage_adp

def main():
    print("="*70)
    print("CALIBRATION DRIFT PATTERNS")
    print("="*70)
    
    patterns = [
        ('linear_bias', 0.3),
        ('linear_bias', 0.5),
        ('linear_bias', 1.0),
        ('scale_drift', 0.3),
        ('exponential', 0.5),
        ('stepwise', 0.8)
    ]
    
    results = []
    
    for drift_type, mag in patterns:
        print(f"\n{drift_type} (mag={mag})...")
        cov_std, cov_adp = test_one(drift_type, mag)
        
        results.append({
            'drift': drift_type,
            'mag': mag,
            'std': cov_std,
            'adp': cov_adp,
            'delta': cov_adp - cov_std
        })
        
        print(f"  Std: {cov_std:.1%}, Adp: {cov_adp:.1%}, Δ: {(cov_adp-cov_std)*100:+.1f}pp")
    
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'='*70}")
    for r in results:
        print(f"{r['drift']:<15} {r['mag']:.1f}  {r['std']:.1%}  {r['adp']:.1%}  {r['delta']*100:+.1f}pp")
    print(f"{'='*70}\n")
    
    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('calibration_results.csv', index=False)
    print("✓ Saved: calibration_results.csv")

if __name__ == '__main__':
    main()
