# How to Run the Real Estate Sentinel App

## Quick Start

1. **Navigate to the project directory:**
   ```bash
   cd /Users/qwang/Desktop/ai-house-agent
   ```

2. **Install dependencies (if not already installed):**
   ```bash
   pip install streamlit pandas numpy plotly pyyaml fredapi yfinance
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the app:**
   - The app will automatically open in your browser
   - Or navigate to: `http://localhost:8501`

## App Structure

- **Main Entry Point:** `streamlit_app.py` - Landing page
- **Pages:** 
  - `pages/01_principle_paydown_view.py` - Principle Paydown Analysis module

## Navigation

- Use the sidebar menu (☰ icon) to navigate between pages
- Or click on page names in the top navigation bar
- The "Principle Paydown Analysis" page contains all the financial analysis tools

## Configuration

Make sure your configuration files are set up:
- `config/privacy/user_profile.yaml` - Personal financial data (GitIgnored)
- `config/constants.yaml` - Shared constants and thresholds

## Troubleshooting

### Import Errors
If you see import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Module Not Found
If modules aren't found, check that you're running from the project root directory.

### Config File Errors
If config files aren't loading, verify they exist in the `config/` directory.

