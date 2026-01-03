"""
Compare adaptive conformal to other baseline methods
"""
import torch
import torch.nn as nn
import numpy as np
from src.data.real_cmapss_loader import prepare_real_cmapss_data
from src.models.sensor_lstm import SensorLSTM

def apply_drift(y, magnitude=0.5):
    """Linear bias drift"""
    n = len(y)
    time = torch.linspace(0, 1, n)
    return y + magnitude * time

def test_baseline_methods():
    print("="*70)
    print("BASELINE COMPARISONS")
    print("="*70)
    
    train_loader, val_loader, test_loader, _ = prepare_real_cmapss_data()
    device = torch.device('cpu')
    input_dim = next(iter(train_loader))[0].shape[2]
    
    model = SensorLSTM(input_dim, 64, 2, 0.2).to(device)
    model.load_state_dict(torch.load('best_real_sensor_model.pt'))
    
    X_test_all, y_test_all = [], []
    for X, y in test_loader:
        X_test_all.append(X)
        y_test_all.append(y)
    
    X_test = torch.cat(X_test_all, dim=0).to(device)
    y_test = torch.cat(y_test_all, dim=0).to(device)
    
    y_test_drifted = apply_drift(y_test, 0.5)
    
    n_cal = int(0.15 * len(X_test))
    X_cal = X_test[:n_cal]
    y_cal = y_test_drifted[:n_cal]
    X_eval = X_test[n_cal:]
    y_eval = y_test_drifted[n_cal:]
    
    # Get calibration quantile
    model.eval()
    with torch.no_grad():
        y_pred_cal = model(X_cal).squeeze()
    scores = torch.abs(y_cal - y_pred_cal)
    alpha = 0.1
    q_level = np.ceil((len(scores)+1)*(1-alpha))/len(scores)
    quantile = torch.quantile(scores, q_level)
    
    results = []
    
    # Baseline 1: Standard Conformal (No Adaptation)
    print("\n1. Standard Conformal...")
    with torch.no_grad():
        y_pred = model(X_eval).squeeze()
    covered = (y_eval >= y_pred - quantile) & (y_eval <= y_pred + quantile)
    coverage_std = covered.float().mean().item()
    
    results.append({
        'method': 'Standard Conformal',
        'coverage': coverage_std,
        'description': 'Fixed quantile, no adaptation'
    })
    
    # Baseline 2: Online Learning (Continuous Updates)
    print("2. Online Learning...")
    model_online = SensorLSTM(input_dim, 64, 2, 0.2).to(device)
    model_online.load_state_dict(torch.load('best_real_sensor_model.pt'))
    optimizer = torch.optim.Adam(model_online.parameters(), lr=1e-5)
    
    preds = []
    batch_size = 50
    
    for i in range(0, len(X_eval), batch_size):
        X_batch = X_eval[i:i+batch_size]
        y_batch = y_eval[i:i+batch_size]
        
        # Predict
        model_online.eval()
        with torch.no_grad():
            y_pred_batch = model_online(X_batch).squeeze()
        preds.append(y_pred_batch)
        
        # Update model continuously
        model_online.train()
        optimizer.zero_grad()
        pred_train = model_online(X_batch).squeeze()
        loss = nn.MSELoss()(pred_train, y_batch)
        loss.backward()
        optimizer.step()
    
    preds = torch.cat(preds)
    covered_online = (y_eval >= preds - quantile) & (y_eval <= preds + quantile)
    coverage_online = covered_online.float().mean().item()
    
    results.append({
        'method': 'Online Learning',
        'coverage': coverage_online,
        'description': 'Continuous model updates, fixed quantile'
    })
    
    # Baseline 3: Sliding Window Recalibration
    print("3. Sliding Window Recalibration...")
    preds, lowers, uppers = [], [], []
    window_size = 100
    
    for i in range(0, len(X_eval), batch_size):
        X_batch = X_eval[i:i+batch_size]
        y_batch = y_eval[i:i+batch_size]
        
        with torch.no_grad():
            y_pred_batch = model(X_batch).squeeze()
        
        # Recalibrate quantile on recent window
        if i >= window_size:
            window_start = max(0, i - window_size)
            X_window = X_eval[window_start:i]
            y_window = y_eval[window_start:i]
            
            with torch.no_grad():
                y_pred_window = model(X_window).squeeze()
            scores_window = torch.abs(y_window - y_pred_window)
            quantile_window = torch.quantile(scores_window, q_level)
        else:
            quantile_window = quantile
        
        preds.append(y_pred_batch)
        lowers.append(y_pred_batch - quantile_window)
        uppers.append(y_pred_batch + quantile_window)
    
    preds = torch.cat(preds)
    lowers = torch.cat(lowers)
    uppers = torch.cat(uppers)
    
    covered_window = (y_eval >= lowers) & (y_eval <= uppers)
    coverage_window = covered_window.float().mean().item()
    
    results.append({
        'method': 'Sliding Window',
        'coverage': coverage_window,
        'description': 'Quantile recalibration only'
    })
    
    # Method 4: Adaptive Conformal (Ours)
    print("4. Adaptive Conformal (Ours)...")
    model_adp = SensorLSTM(input_dim, 64, 2, 0.2).to(device)
    model_adp.load_state_dict(torch.load('best_real_sensor_model.pt'))
    optimizer_adp = torch.optim.Adam(model_adp.parameters(), lr=1e-4)
    
    preds, lowers, uppers = [], [], []
    residual_buffer, X_buffer, y_buffer = [], [], []
    current_quantile = quantile
    
    for i in range(0, len(X_eval), batch_size):
        X_batch = X_eval[i:i+batch_size]
        y_batch = y_eval[i:i+batch_size]
        
        model_adp.eval()
        with torch.no_grad():
            y_pred_batch = model_adp(X_batch).squeeze()
        
        preds.append(y_pred_batch)
        lowers.append(y_pred_batch - current_quantile)
        uppers.append(y_pred_batch + current_quantile)
        
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
                optimizer_adp.zero_grad()
                pred = model_adp(X_recent).squeeze()
                loss = nn.MSELoss()(pred, y_recent)
                loss.backward()
                optimizer_adp.step()
            
            model_adp.eval()
            with torch.no_grad():
                y_pred_new = model_adp(X_recent).squeeze()
            new_residuals = torch.abs(y_recent - y_pred_new)
            residual_buffer = new_residuals.cpu().tolist()
            
            if len(residual_buffer) >= 50:
                recent_residuals = torch.tensor(residual_buffer).to(device)
                new_quantile = torch.quantile(recent_residuals, q_level)
                current_quantile = 0.5 * current_quantile + 0.5 * new_quantile
    
    preds = torch.cat(preds)
    lowers = torch.cat(lowers)
    uppers = torch.cat(uppers)
    
    covered_adp = (y_eval >= lowers) & (y_eval <= uppers)
    coverage_adp = covered_adp.float().mean().item()
    
    results.append({
        'method': 'Adaptive Conformal (Ours)',
        'coverage': coverage_adp,
        'description': 'Model + quantile adaptation'
    })
    
    # Print results
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON RESULTS:")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Coverage':<12} {'vs Standard':<15}")
    print("-"*70)
    
    for r in results:
        improvement = (r['coverage'] - coverage_std) * 100
        print(f"{r['method']:<30} {r['coverage']:<11.1%} {improvement:>+13.1f}pp")
    
    print(f"{'='*70}\n")
    
    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('baseline_comparison.csv', index=False)
    print("✓ Saved: baseline_comparison.csv")
    
    return results

if __name__ == '__main__':
    test_baseline_methods()
