import sys
from pathlib import Path
import yaml
import pandas as pd

# Add src to sys.path
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path / "src"))

from sentinel.principle_paydown.principle_payment_calculator import RealEstateSentinel

def verify_sensitivity():
    print("Starting verification (Dynamic Balance)...")
    
    # 1. Baseline Run (No extra pay)
    sentinel = RealEstateSentinel(
        user_config_path=base_path / "config" / "privacy" / "user_profile.yaml",
        constants_config_path=base_path / "config" / "constants.yaml",
        override_inputs={'principle_bulk_to_pay': 0.0}
    )
    results_base = sentinel.run_analysis()
    df_base = results_base['solvency']['sensitivity_table']
    
    # Extract Dec 2026 balance from baseline
    row_base_future = df_base[df_base['Scenario'] == "Sell Dec 2026"].iloc[0]
    bal_base_future = row_base_future['My Mortgage Balance']
    print(f"Baseline Future Balance: ${bal_base_future:,.0f}")
    
    # 2. Simulated Run ($100k Paydown in Jan 2026)
    sim_pay = 100000.0
    sentinel_sim = RealEstateSentinel(
        user_config_path=base_path / "config" / "privacy" / "user_profile.yaml",
        constants_config_path=base_path / "config" / "constants.yaml",
        override_inputs={
            'principle_bulk_to_pay': sim_pay,
            'principle_bulk_paydown_date': datetime(2026, 1, 15).date()
        }
    )
    results_sim = sentinel_sim.run_analysis()
    df_sim = results_sim['solvency']['sensitivity_table']
    
    # Extract Dec 2026 balance from sim
    row_sim_future = df_sim[df_sim['Scenario'] == "Sell Dec 2026"].iloc[0]
    bal_sim_future = row_sim_future['My Mortgage Balance']
    print(f"Simulated Future Balance: ${bal_sim_future:,.0f}")
    
    # 3. Validation
    diff = bal_base_future - bal_sim_future
    print(f"Difference: ${diff:,.0f}")
    
    if diff >= sim_pay:
        print("PASS: Future balance reduced by at least the paydown amount.")
    else:
        print("FAIL: Future balance did not decrease enough.")
        
    # Check "Sell Now" balance shouldn't change if paydate is in future
    row_sim_now = df_sim[df_sim['Scenario'] == "Sell Now"].iloc[0]
    bal_sim_now = row_sim_now['My Mortgage Balance']
    
    # Compare with base "Sell Now" (should be identical)
    row_base_now = df_base[df_base['Scenario'] == "Sell Now"].iloc[0]
    bal_base_now = row_base_now['My Mortgage Balance']
    
    if abs(bal_sim_now - bal_base_now) < 5: # float tolerance
        print(f"PASS: 'Sell Now' balance unaffected by future paydown (${bal_sim_now:,.0f}).")
    else:
         print(f"FAIL: 'Sell Now' balance changed! Base: {bal_base_now}, Sim: {bal_sim_now}")

if __name__ == "__main__":
    from datetime import datetime
    verify_sensitivity()
