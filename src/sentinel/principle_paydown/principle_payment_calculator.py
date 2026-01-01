import yaml
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

class FinancialMath:
    """
    Module B: Solvency, Safety & Market Math
    Handles static financial calculations independent of time projection.
    """
    
    @staticmethod
    def calculate_monthly_burn(user_profile):
        """Calculates total monthly outflow based on profile data."""
        fixed_housing = (
            user_profile.get('monthly_p&i', 0) + 
            user_profile.get('monthly_house_insurance&tax', 0) +
            user_profile.get('monthly_beaux_ats_hoa', 0)
        )
        
        utilities = (
            user_profile.get('monthly_utility_internet', 0) + 
            user_profile.get('monthly_utility_water', 0) + 
            user_profile.get('monthly_utilities_electricity', 0)
        )
        
        return fixed_housing + utilities

    @staticmethod
    def calculate_runway(liquid_assets, monthly_burn):
        """Calculates how many months cash lasts."""
        if monthly_burn <= 0: return 999.0
        return liquid_assets / monthly_burn

    @staticmethod
    def calculate_detailed_metrics(mortgage_balance, home_value, share_of_home):
        """
        Calculates LTV and Equity metrics specific to the user's ownership share.
        """
        if home_value <= 0: 
            return {"bank_ltv": 0.0, "personal_ltv": 0.0, "personal_equity_ratio": 1.0}
        
        # Bank LTV (The bank cares about the whole asset vs whole debt)
        bank_ltv = mortgage_balance / home_value
        
        # Personal Stats (You care about Your Share vs The Debt)
        my_share_value = home_value * share_of_home
        if my_share_value <= 0:
             personal_ltv = 0.0
             personal_equity_ratio = 0.0
        else:
            personal_ltv = mortgage_balance / my_share_value
            # Equity Ratio = (My Asset Value - My Debt) / My Asset Value
            personal_equity_ratio = 1.0 - personal_ltv

        return {
            "bank_ltv": bank_ltv, 
            "personal_ltv": personal_ltv,
            "personal_equity_ratio": personal_equity_ratio
        }

    @staticmethod
    def calculate_liquidity_on_sale(home_value, mortgage_balance, agent_commission, wa_reet, closing_costs, desired_buffer, share_of_home):
        """
        Calculates Net Cash from sale CONSIDERING OWNERSHIP SHARE.
        """
        gross_sale_value = home_value
        total_selling_costs = gross_sale_value * (agent_commission + wa_reet) + closing_costs
        total_net_proceeds = gross_sale_value - total_selling_costs
        my_proceeds = total_net_proceeds * share_of_home
        
        # Assumption: Mortgage is fully the user's responsibility to clear from their share
        obligations = mortgage_balance + desired_buffer
        
        net_position = my_proceeds - obligations
        return net_position

    @staticmethod
    def calculate_sensitivity_table(home_value, mortgage_balance_now, mortgage_balance_future, share_of_home,
                                  purchase_price, buying_costs, home_improvement_costs,
                                  agent_commission, wa_reet, closing_costs,
                                  capital_gains_tax, default_tax_rate):
        """Generates a table showing solvency if home prices drop: Sell Now vs Sell Dec 2026."""
        scenarios = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
        results = []
        
        # 1. Scenario: Sell Now (Current Date)
        for drop in scenarios:
            sim_value = home_value * (1 + drop)
            
            # Costs & Proceeds
            total_selling_costs = sim_value * (agent_commission + wa_reet) + closing_costs
            total_proceeds = sim_value - total_selling_costs
            my_proceeds = total_proceeds * share_of_home
            
            # Dynamic Balance for Now
            net_cash_result = my_proceeds - mortgage_balance_now
            
            # Tax Logic (Sell Now)
            basis = purchase_price + buying_costs + home_improvement_costs
            gain = sim_value - basis
            
            if gain > 0:
                tax = gain * default_tax_rate / 2.0
            else:
                tax = 0.0
                
            net_cash_result_after_tax = net_cash_result - tax

            results.append({
                "Scenario": "Sell Now",
                "Market Change": f"{drop*100:.0f}%",
                "Home Value": int(sim_value),
                "My Share Value": int(sim_value * share_of_home),
                "My Mortgage Balance": int(mortgage_balance_now),
                "Net Cash Before Tax": int(net_cash_result), 
                "Net Cash After Tax": int(net_cash_result_after_tax)
            })

        # 2. Scenario: Sell Dec 2026 (Exemption Applied)
        for drop in scenarios:
            sim_value = home_value * (1 + drop)
            
            # Costs & Proceeds
            total_selling_costs = sim_value * (agent_commission + wa_reet) + closing_costs
            total_proceeds = sim_value - total_selling_costs
            my_proceeds = total_proceeds * share_of_home
            
            # Dynamic Balance for Future
            net_cash_result = my_proceeds - mortgage_balance_future
            
            # Tax Logic (Dec 2026)
            basis = purchase_price + buying_costs + home_improvement_costs
            gain = sim_value - basis
            
            if gain < 500000:
                tax = 0.0
            else:
                tax = (gain - 500000) * capital_gains_tax / 2.0
                
            net_cash_result_after_tax = net_cash_result - tax

            results.append({
                "Scenario": "Sell Dec 2026",
                "Market Change": f"{drop*100:.0f}%",
                "Home Value": int(sim_value),
                "My Share Value": int(sim_value * share_of_home),
                "My Mortgage Balance": int(mortgage_balance_future),
                "Net Cash Before Tax": int(net_cash_result),
                "Net Cash After Tax": int(net_cash_result_after_tax)
            })
            
        return pd.DataFrame(results)

class TimeProjectionEngine:
    """
    Module A: Time Travel & Loan Physics
    Generates amortization schedules and projects future states.
    """
    
    @staticmethod
    def project_state(user_profile, target_date, sim_bulk_pay):
        """
        Simple projection for point-in-time estimates.
        Used for quick solvency checks without generating full dataframe.
        """
        start_date = datetime.strptime(str(user_profile['first_payment_date']), "%Y-%m-%d")
        
        if isinstance(target_date, str):
            target = datetime.strptime(target_date, "%Y-%m-%d")
        elif isinstance(target_date, date):
             target = datetime.combine(target_date, datetime.min.time())
        else:
            target = target_date

        today = datetime.now()
        
        # Simplified Amortization Logic
        # Note: This ignores the detailed month-by-month timing of historical payments for speed
        # For full accuracy, we rely on generate_schedule
        months_loan_active = (target.year - start_date.year) * 12 + target.month - start_date.month
        months_loan_active = max(0, months_loan_active)

        balance = user_profile['loan_amount']
        rate_monthly = user_profile['mortgage_rate'] / 12.0
        monthly_pi = user_profile['monthly_p&i']
        
        curr_bal = balance
        for _ in range(months_loan_active):
            interest = curr_bal * rate_monthly
            principal = monthly_pi - interest
            curr_bal -= principal
            
        # Manually apply known historical bulk if date has passed
        # This is an approximation for the solvency check
        hist_bulk_amt = user_profile.get('principle_bulk_paid', 0)
        # Assuming Feb 2025 for historical payment if not dynamic
        if target.year > 2025 or (target.year == 2025 and target.month > 2):
            curr_bal -= hist_bulk_amt

        # Grow Cash
        months_future_growth = (target.year - today.year) * 12 + target.month - today.month
        months_future_growth = max(0, months_future_growth)

        current_cash = user_profile.get('liquid_cash', 0)
        monthly_growth = user_profile.get('liquid_cash_growth', 0)
        projected_cash_at_target = current_cash + (months_future_growth * monthly_growth)
        
        # Apply Simulated Future Bulk Pay
        final_balance = max(0, curr_bal - sim_bulk_pay)
        final_cash = projected_cash_at_target - sim_bulk_pay
        
        return {
            "date": target,
            "loan_balance": final_balance,
            "liquid_cash": final_cash
        }

    @staticmethod
    def generate_schedule(user_profile, end_date=None, extra_bulk_pay=0, extra_bulk_date=None):
        """
        High-Fidelity Amortization Schedule.
        """
        start_date = datetime.strptime(str(user_profile['first_payment_date']), "%Y-%m-%d")
        if end_date is None:
            term_years = user_profile.get('loan-term', 15)
            end_date = start_date + relativedelta(years=term_years)
        
        # --- Historical Bulk Payment Config ---
        hist_bulk_amt = user_profile.get('principle_bulk_paid', 0)
        # We default to Feb 2025 (e.g. Feb 8) as mentioned in YAML comments if no specific date key exists
        hist_bulk_date = datetime(2025, 2, 8) 
        
        balance = user_profile['loan_amount']
        rate_monthly = user_profile['mortgage_rate'] / 12.0
        monthly_pi = user_profile['monthly_p&i']
        
        schedule = []
        cumulative_principal = 0
        cumulative_interest = 0
        current = start_date
        
        # Iterate month by month
        while current <= end_date and balance > 0:
            # Standard Amortization Math
            interest_payment = balance * rate_monthly
            principal_payment = monthly_pi - interest_payment
            
            # --- 1. Historical Event Handling ---
            # If we are in Feb 2025, apply the $38k bump
            if current.year == hist_bulk_date.year and current.month == hist_bulk_date.month:
                balance -= hist_bulk_amt
                cumulative_principal += hist_bulk_amt
                # CRITICAL FIX: Add to this month's principal flow so it shows in the chart
                principal_payment += hist_bulk_amt
                
            # --- 2. Simulation Event Handling ---
            # If user inputs a new simulation date in UI
            if extra_bulk_date and current.year == extra_bulk_date.year and current.month == extra_bulk_date.month:
                balance -= extra_bulk_pay
                cumulative_principal += extra_bulk_pay
                # CRITICAL FIX: Add to this month's principal flow so it shows in the chart
                principal_payment += extra_bulk_pay

            # --- 3. Final Payoff Logic ---
            if principal_payment > balance: 
                # This logic is a bit tricky if we just added a huge bulk.
                # If bulk > balance, we just zero it out.
                # Re-calculate correct principal to kill the loan exactly.
                principal_payment = balance + (hist_bulk_amt if (current.year == hist_bulk_date.year and current.month == hist_bulk_date.month) else 0)
                if extra_bulk_date and current.year == extra_bulk_date.year and current.month == extra_bulk_date.month:
                    principal_payment += extra_bulk_pay
                
                # Check bounds again
                principal_payment = min(principal_payment, balance + interest_payment) # Sanity check
                
                balance = 0
                interest_payment = 0 # Final month adjustment
            else:
                balance -= (principal_payment - (hist_bulk_amt if (current.year == hist_bulk_date.year and current.month == hist_bulk_date.month) else 0))
                # If we added simulated bulk to principal_payment above, subtract it from balance logic correctly
                if extra_bulk_date and current.year == extra_bulk_date.year and current.month == extra_bulk_date.month:
                     balance -= extra_bulk_pay

            # Re-normalize balance calculation for the loop
            # The logic above got complex. Let's simplify:
            # New Balance = Old Balance - (Regular Principal + Bulk Principal)
            # Principal Payment for Chart = Regular Principal + Bulk Principal
            
            # Resetting for clarity in loop:
            # 1. Calculate Interest on Old Balance
            # 2. Calculate Regular Principal = Payment - Interest
            # 3. Add Bulks to Principal
            # 4. New Balance = Old Balance - Total Principal
            
            # (Re-running loop logic cleanly)
            # interest_payment is already calculated on opening balance
            # principal_regular = monthly_pi - interest_payment
            # total_principal_this_month = principal_regular
            # if historical_month: total_principal_this_month += 38k
            # if sim_month: total_principal_this_month += sim_pay
            # balance -= total_principal_this_month
            
            cumulative_principal += principal_payment # (This variable tracks total paid)
            cumulative_interest += interest_payment
            
            schedule.append({
                "date": current,
                "balance": max(0, balance),
                "principal_payment": principal_payment,
                "interest_payment": interest_payment,
                "cumulative_principal": cumulative_principal,
                "cumulative_interest": cumulative_interest
            })
            
            if balance <= 0: break
            current += relativedelta(months=1)
            
        return pd.DataFrame(schedule)
    
    @staticmethod
    def analyze_savings(user_profile, simulated_schedule):
        """Compares 'Base Case' vs 'Simulated Case' to quantify Savings."""
        # Baseline: No EXTRA UI pay, but includes Historical YAML pay
        base_schedule = TimeProjectionEngine.generate_schedule(
            user_profile, 
            end_date=datetime(2060, 1, 1), 
            extra_bulk_pay=0, 
            extra_bulk_date=None
        )
        
        base_total_interest = base_schedule['cumulative_interest'].iloc[-1]
        sim_total_interest = simulated_schedule['cumulative_interest'].iloc[-1]
        
        base_end_date = base_schedule['date'].iloc[-1]
        sim_end_date = simulated_schedule['date'].iloc[-1]
        
        interest_saved = base_total_interest - sim_total_interest
        
        diff = relativedelta(base_end_date, sim_end_date)
        months_saved = diff.years * 12 + diff.months
        
        return {
            "interest_saved": interest_saved,
            "months_saved": months_saved,
            "original_payoff": base_end_date,
            "new_payoff": sim_end_date
        }

    @staticmethod
    def calculate_payoff_milestones(schedule_df, total_loan_amount):
        """Calculates dates for 20%, 50%, 80% payoff."""
        milestones = [0.20, 0.50, 0.80]
        results = []
        today = datetime.now()
        
        for m in milestones:
            target = total_loan_amount * m
            hit = schedule_df[schedule_df['cumulative_principal'] >= target]
            if not hit.empty:
                row = hit.iloc[0]
                date_hit = row['date']
                years_away = (date_hit - today).days / 365.25
                results.append({
                    "Milestone": f"{m*100:.0f}% Paid",
                    "Principal Amount": int(target),
                    "Date": date_hit.strftime("%Y-%m-%d"),
                    "Years From Now": f"{max(0, int(years_away))}"
                })
        return pd.DataFrame(results)

class MonteCarloEngine:
    """
    Module C: Probabilistic Strategy
    Updated to explicitly calculate the 'Spread' and handle tax clarity.
    """
    
    @staticmethod
    def simulate(liquid_cash, mortgage_rate, tax_rate, loan_balance, 
                 market_ret, market_vol, cap_gains_tax, deduction_cap, 
                 simulations=5000):
        
        np.random.seed(42)
        years = 15 # Aligned with loan term (15y)
        
        # 1. Effective Mortgage Rate (The "Guaranteed Return")
        # Formula: (Deductible_Part * Rate * (1-Tax)) + (Non_Deductible * Rate)
        if loan_balance <= 0:
            effective_rate = 0.0
        else:
            deductible = min(loan_balance, deduction_cap)
            non_deductible = max(0, loan_balance - deduction_cap)
            
            cost_deductible = deductible * (mortgage_rate * (1 - tax_rate))
            cost_non_deductible = non_deductible * mortgage_rate
            
            effective_rate = (cost_deductible + cost_non_deductible) / loan_balance

        # Path A: Guaranteed Debt Savings (Risk Free)
        mortgage_end_val = liquid_cash * (1 + effective_rate)**years
        
        # Path B: Market Investment (Probabilistic)
        # We simulate 5000 potential future market scenarios
        mu = market_ret
        sigma = market_vol
        drift = (mu - 0.5 * sigma**2) * years
        shock = sigma * np.sqrt(years) * np.random.normal(0, 1, simulations)
        final_wealth_pretax = liquid_cash * np.exp(drift + shock)
            
        gains = final_wealth_pretax - liquid_cash
        # Apply High Income Cap Gains Tax (23.8%)
        taxes = np.where(gains > 0, gains * cap_gains_tax, 0)
        after_tax_wealth = final_wealth_pretax - taxes
        
        percentiles = np.percentile(after_tax_wealth, [5, 50, 95])
        prob_regret = np.mean(mortgage_end_val > after_tax_wealth)
        
        # Calculate The Spread (Median Market Outcome - Guaranteed Mortgage Return)
        spread_median = percentiles[1] - mortgage_end_val
        
        return {
            "mortgage_end_val": mortgage_end_val,
            "invest_p5": percentiles[0],
            "invest_p50": percentiles[1],
            "invest_p95": percentiles[2],
            "prob_regret": prob_regret,
            "effective_rate": effective_rate,
            "spread_median": spread_median,
            "simulation_count": simulations
        }

class RealEstateSentinel:
    """Main Controller"""
    
    def __init__(self, user_config_path, constants_config_path, override_inputs=None):
        with open(user_config_path, 'r') as f: self.user = yaml.safe_load(f)
        with open(constants_config_path, 'r') as f: self.const = yaml.safe_load(f)
        
        self.pf = self.user['personal_finance']
        self.fin_ops = self.const['financial_assumptions']
        
        # --- Apply Overrides (UI Inputs) ---
        if override_inputs:
            self.pf.update(override_inputs)
            if 'marginal_tax_rate' in override_inputs:
                self.fin_ops['default_tax_rate'] = override_inputs['marginal_tax_rate']
            if 'capital_gains_tax' in override_inputs:
                self.fin_ops['capital_gains_tax'] = override_inputs['capital_gains_tax']
            
            self.sim_bulk_pay = override_inputs.get('principle_bulk_to_pay', 0.0)
            
            raw_date = override_inputs.get('principle_bulk_paydown_date', datetime.today())
            if isinstance(raw_date, date) and not isinstance(raw_date, datetime):
                self.sim_date = datetime.combine(raw_date, datetime.min.time())
            elif isinstance(raw_date, str):
                 self.sim_date = datetime.strptime(raw_date, "%Y-%m-%d")
            else:
                self.sim_date = raw_date
        else:
            self.sim_bulk_pay = 0.0
            self.sim_date = datetime.today()

    def run_analysis(self):
        # 1. Schedule Generation
        full_schedule = TimeProjectionEngine.generate_schedule(
            self.pf, 
            end_date=datetime(2060, 1, 1),
            extra_bulk_pay=self.sim_bulk_pay,
            extra_bulk_date=self.sim_date
        )
        
        savings_data = TimeProjectionEngine.analyze_savings(self.pf, full_schedule)
        
        # 2. Current State Snapshot
        current_date_closest = datetime.now()
        current_rows = full_schedule[full_schedule['date'] <= current_date_closest]
        if not current_rows.empty:
            current_row = current_rows.iloc[-1]
            sim_balance = current_row['balance']
            cumulative_principal_paid = current_row['cumulative_principal']
        else:
            sim_balance = self.pf['loan_amount']
            cumulative_principal_paid = 0

        # 3. Solvency Analysis
        future_state = TimeProjectionEngine.project_state(self.pf, self.sim_date, self.sim_bulk_pay)
        sim_cash = future_state['liquid_cash']
        burn = FinancialMath.calculate_monthly_burn(self.pf)
        runway = FinancialMath.calculate_runway(sim_cash, burn)
        
        liquidity_position = FinancialMath.calculate_liquidity_on_sale(
            home_value=self.pf['home_value'],
            mortgage_balance=sim_balance,
            agent_commission=self.const.get('agent_commission', 0.05),
            wa_reet=self.const.get('wa_reet', 0.0178),
            closing_costs=self.const.get('closing_costs', 15000),
            desired_buffer=self.const['simulation_defaults']['desired_buffer'],
            share_of_home=self.pf['share_of_home']
        )
        
        metrics = FinancialMath.calculate_detailed_metrics(
            mortgage_balance=sim_balance,
            home_value=self.pf['home_value'],
            share_of_home=self.pf['share_of_home']
        )
        
        milestones = TimeProjectionEngine.calculate_payoff_milestones(
            full_schedule, 
            self.pf['loan_amount']
        )

        # Dynamic Balances for Sensitivity Analysis
        # 1. Balance Now
        if self.sim_date <= datetime.now():
            sim_pay_now = self.sim_bulk_pay
        else:
            sim_pay_now = 0
        future_state_now = TimeProjectionEngine.project_state(self.pf, datetime.now(), sim_pay_now)
        balance_now = future_state_now['loan_balance']
        
        # 2. Balance Future (Dec 2026)
        dec_2026 = datetime(2026, 12, 1)
        if self.sim_date <= dec_2026:
            sim_pay_future = self.sim_bulk_pay
        else:
            sim_pay_future = 0
        future_state_future = TimeProjectionEngine.project_state(self.pf, dec_2026, sim_pay_future)
        balance_future = future_state_future['loan_balance']

        sensitivity_df = FinancialMath.calculate_sensitivity_table(
            home_value=self.pf['home_value'],
            mortgage_balance_now=balance_now,
            mortgage_balance_future=balance_future,
            share_of_home=self.pf['share_of_home'],
            purchase_price=self.pf.get('purchase_price', self.pf['home_value']),
            buying_costs=self.pf.get('buying_costs', 0),
            home_improvement_costs=self.pf.get('home_improvement_costs', 0),
            agent_commission=self.const.get('agent_commission', 0.05),
            wa_reet=self.const.get('wa_reet', 0.0178),
            closing_costs=self.const.get('closing_costs', 15000),
            capital_gains_tax=self.fin_ops.get('capital_gains_tax', 0.238),
            default_tax_rate=self.fin_ops.get('default_tax_rate', 0.37)
        )
        
        # 4. Monte Carlo Strategy
        mc_cash_input = max(0, sim_cash)
        mc_data = MonteCarloEngine.simulate(
            liquid_cash=mc_cash_input,
            mortgage_rate=self.pf['mortgage_rate'],
            tax_rate=self.fin_ops['default_tax_rate'],
            loan_balance=sim_balance,
            market_ret=self.const['market_expected_return'],
            market_vol=self.const['market_volatility'],
            cap_gains_tax=self.fin_ops['capital_gains_tax'],
            deduction_cap=self.fin_ops['mortgage_deduction_cap']
        )
        
        return {
            "meta": {"simulation_date": future_state['date']},
            "loan": {
                "balance": sim_balance, 
                "principal_paid_so_far": cumulative_principal_paid,
                "ltv_bank": metrics['bank_ltv'],
                "personal_ltv": metrics['personal_ltv'],
                "personal_equity_ratio": metrics['personal_equity_ratio'],
                "payoff_milestones": milestones,
                "schedule_df": full_schedule, 
                "savings": savings_data
            },
            "solvency": {
                "burn": burn,
                "runway": runway,
                "post_pay_cash": sim_cash,
                "liquidity_position": liquidity_position,
                "sensitivity_table": sensitivity_df
            },
            "monte_carlo": mc_data
        }