# Installing Dependencies

If you're getting `ModuleNotFoundError` errors, install all required dependencies:

## Quick Install (All at Once)

```bash
cd /Users/qwang/Desktop/ai-house-agent
python3 -m pip install -r requirements.txt
```

Or install individually:

```bash
python3 -m pip install streamlit pandas numpy plotly pyyaml fredapi yfinance
```

## Verify Installation

After installing, verify all packages are available:

```bash
python3 -c "import streamlit, pandas, numpy, plotly, yaml, fredapi, yfinance; print('✅ All packages installed!')"
```

## If Streamlit Uses Different Python Environment

If you're still getting errors after installing, Streamlit might be using a different Python environment. Try:

1. **Check which Python Streamlit is using:**
   ```bash
   which streamlit
   python3 -m streamlit --version
   ```

2. **Install dependencies using the same Python:**
   ```bash
   python3 -m pip install -r requirements.txt --user
   ```

3. **Or use a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```

## Troubleshooting

- **"command not found: pip"** → Use `python3 -m pip` instead of `pip`
- **Permission errors** → Add `--user` flag: `python3 -m pip install --user package_name`
- **Still getting errors after install** → Restart the Streamlit app (Ctrl+C and run again)

