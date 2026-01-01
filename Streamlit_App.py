"""
Real Estate Sentinel - Finance Suite App
Main entry point for the Streamlit application.

This app provides tools for real estate financial decision-making,
including mortgage paydown analysis, investment comparisons, and equity tracking.
"""

import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Real Estate Sentinel",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main landing page
st.title("🛡️ Real Estate Sentinel")
st.markdown("---")

st.markdown("""
### Welcome to Real Estate Sentinel

A comprehensive financial analysis tool for making informed decisions about your real estate investments.

**Available Modules:**
- **Principle Paydown Analysis** - Analyze whether to pay down your mortgage faster or invest
- **Equity Milestone Tracking** - Visualize your path to key equity milestones
- **Market Context** - Real-time market data and benchmarks

### Getting Started

Use the sidebar navigation to access different modules, or click on a page name above.

**Note:** Personal financial data is loaded from `config/privacy/user_profile.yaml`.
Make sure your configuration is set up before running analyses.
""")

st.markdown("---")

# Display quick info section
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Status", "Ready", help="Application is ready to use")

with col2:
    st.metric("Modules", "1 Active", help="Number of available analysis modules")

with col3:
    st.metric("Configuration", "Loaded", help="Config files are loaded")

st.markdown("---")
st.caption("💡 **Tip:** Navigate using the sidebar menu or click on page names above to access different analysis tools.")
