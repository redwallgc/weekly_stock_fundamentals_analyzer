# app.py - Streamlit Application Code (V4.0 - Multi-User Authentication)
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import io

# --- 1. CONFIGURATION AND AUTHENTICATION ---

# Set a persistent state variable for login status
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_password():
    """Returns True if the user enters a correct password from the list of users."""
    # Check if a 'users' section exists in the secrets file
    if 'users' not in st.secrets:
        st.error("Authentication credentials not configured. Please check Streamlit secrets.")
        return False
        
    def password_entered():
        """Checks whether a password entered is correct."""
        input_username = st.session_state["username"]
        input_password = st.session_state["password"]
        
        # Iterate through all configured user sections in st.secrets
        # We look for any key that starts with 'user_'
        for user_key in st.secrets.keys():
            if user_key.startswith('user_'):
                user_credentials = st.secrets[user_key]
                
                if (input_username == user_credentials.get("username") and 
                    input_password == user_credentials.get("password")):
                    st.session_state["authenticated"] = True
                    del st.session_state["password"]
                    return True

        st.session_state["authenticated"] = False
        return False

    if st.session_state["authenticated"]:
        return True

    # --- Display login form if not authenticated ---
    with st.form("login_form"):
        st.subheader("Login Required to Access Analysis Tool")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.form_submit_button("Log in", on_click=password_entered)

    if 'password' in st.session_state and not st.session_state["authenticated"]:
        st.error("Authentication failed. Please check username and password.")
    
    return st.session_state["authenticated"]


# --- 2. ANALYSIS LOGIC (The core engine) ---

COLUMN_ORDER = [
    'Ticker', 
    'Overall Score',
    'Balance Sheet (Current assets/current liabilities)', 'CR Pass? (>=1)',
    'Income statement (Op. Income/Total Revenue %)', 'OM Pass? (>=15)',
    'P/E Ratio', 'P/E Pass? (>=20)', 
    'P/S Ratio', 'P/S Pass? (<=2)',
    'Cash flow statement (FCF Trend)', 'FCF Pass?',
    'Current Price', 'Target Price', 'Target Pass? (P<T)'
]
SCORING_WEIGHTS = {
    'CR Pass? (>=1)': 2,
    'Target Pass? (P<T)': 2,
    'OM Pass? (>=15)': 1,
    'FCF Pass?': 1,
    'P/E Pass? (>=20)': 1,
    'P/S Pass? (<=2)': 1
}

@st.cache_data(show_spinner="Fetching financial data...")
def run_analysis(ticker_list):
    analysis_results = []
    
    def analyze_ticker(ticker_symbol):
        data = {'Ticker': ticker_symbol}
        for col in COLUMN_ORDER:
            if col != 'Ticker':
                data[col] = 'ERROR'
        score = 0
        
        try:
            # Data Fetching and Metric Calculation Logic
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            bs = ticker.balance_sheet
            cf = ticker.cashflow
            is_stmt = ticker.financials
            
            if not info or bs.empty or cf.empty or is_stmt.empty:
                 raise ValueError(f"No fundamental data found for {ticker_symbol}.")
            
            latest_bs = bs.iloc[:, 0].to_dict()
            latest_is = is_stmt.iloc[:, 0].to_dict()
            
            def safe_get(data_dict, key, default=np.nan):
                return data_dict.get(key, default)
            
            # --- A. Current Ratio ---
            current_assets = safe_get(latest_bs, 'Current Assets')
            current_liabilities = safe_get(latest_bs, 'Current Liabilities')
            current_ratio = current_assets / current_liabilities if current_liabilities and np.isfinite(current_liabilities) and current_liabilities != 0 else np.nan
            cr_pass = np.isfinite(current_ratio) and current_ratio >= 1.0
            
            # --- B. Operating Margin ---
            operating_income = safe_get(latest_is, 'Operating Income')
            total_revenue = safe_get(latest_is, 'Total Revenue')
            operating_margin = (operating_income / total_revenue) * 100 if total_revenue and np.isfinite(total_revenue) and total_revenue != 0 else np.nan
            om_pass = np.isfinite(operating_margin) and operating_margin >= 15.0

            # --- C. P/E and P/S Ratios ---
            pe_ratio = info.get('trailingPE', np.nan)
            ps_ratio = info.get('priceToSales', np.nan) 
            if not np.isfinite(ps_ratio):
                market_cap = info.get('marketCap', np.nan)
                if np.isfinite(market_cap) and np.isfinite(total_revenue) and total_revenue != 0:
                    ps_ratio = market_cap / total_revenue
            pe_pass = np.isfinite(pe_ratio) and pe_ratio >= 20.0
            ps_pass = np.isfinite(ps_ratio) and ps_ratio <= 2.0

            # --- D. FCF Trend ---
            annual_cf = ticker.cashflow.T
            is_increasing = False
            fcf_trend_label = 'Not enough data'
            if len(annual_cf) >= 3:
                cash_metric_name = 'Free Cash Flow' if 'Free Cash Flow' in annual_cf.columns else 'Operating Cash Flow'
                if cash_metric_name in annual_cf.columns:
                    cfs = annual_cf[cash_metric_name].iloc[0:3].fillna(0)
                    is_increasing = (cfs.iloc[0] > cfs.iloc[1]) and (cfs.iloc[1] > cfs.iloc[2])
                    fcf_trend_label = '📈 Increasing' if is_increasing else '📉 Not Consistent'
                else:
                    fcf_trend_label = 'Metric not found'
            fcf_pass = is_increasing
            
            # --- E. Analyst Price Target ---
            target_price = info.get('targetMedianPrice', np.nan)
            current_price = info.get('currentPrice', np.nan)
            target_pass = np.isfinite(target_price) and np.isfinite(current_price) and (current_price < target_price)

            # 4. Calculate Score and Map Values
            pass_fail_statuses = {
                'CR Pass? (>=1)': cr_pass,
                'OM Pass? (>=15)': om_pass,
                'P/E Pass? (>=20)': pe_pass,
                'P/S Pass? (<=2)': ps_pass,
                'FCF Pass?': fcf_pass,
                'Target Pass? (P<T)': target_pass
            }

            for metric, is_passing in pass_fail_statuses.items():
                if is_passing:
                    score += SCORING_WEIGHTS.get(metric, 0)
            
            data['Overall Score'] = score

            data['Balance Sheet (Current assets/current liabilities)'] = f"{current_ratio:.2f}" if np.isfinite(current_ratio) else 'N/A'
            data['CR Pass? (>=1)'] = '✅ PASS' if cr_pass else '❌ FAIL'
            
            data['Income statement (Op. Income/Total Revenue %)'] = f"{operating_margin:.2f}%" if np.isfinite(operating_margin) else 'N/A'
            data['OM Pass? (>=15)'] = '✅ PASS' if om_pass else '❌ FAIL'

            data['P/E Ratio'] = f"{pe_ratio:.2f}" if np.isfinite(pe_ratio) else 'N/A'
            data['P/E Pass? (>=20)'] = '✅ PASS' if pe_pass else '❌ FAIL'
            
            data['P/S Ratio'] = f"{ps_ratio:.2f}" if np.isfinite(ps_ratio) else 'N/A'
            data['P/S Pass? (<=2)'] = '✅ PASS' if ps_pass else '❌ FAIL'
            
            data['Cash flow statement (FCF Trend)'] = fcf_trend_label
            data['FCF Pass?'] = '✅ PASS' if fcf_pass else '❌ FAIL'
            
            data['Current Price'] = f"${current_price:.2f}" if np.isfinite(current_price) else 'N/A'
            data['Target Price'] = f"${target_price:.2f}" if np.isfinite(target_price) else 'N/A'
            
            data['Target Pass? (P<T)'] = '✅ PASS' if target_pass else '❌ FAIL'
                
        except Exception:
            pass
        
        analysis_results.append(data)

    for ticker in ticker_list:
        analyze_ticker(ticker)

    final_df = pd.DataFrame(analysis_results)
    final_df = final_df.sort_values(by='Overall Score', ascending=False)
    final_df = final_df[COLUMN_ORDER]
    
    return final_df

# --- 3. MAIN APPLICATION LOGIC (Requires successful authentication) ---

if check_password():
    st.set_page_config(page_title="Weekly Stock Analysis Tool", layout="wide")
    st.title("📊 Weekly Fundamental Analysis Tool")
    st.markdown("Upload your single-column CSV of stock tickers below to run the analysis.")

    uploaded_file = st.file_uploader(
        "1. Choose a CSV file (single column of tickers)", 
        type=["csv"]
    )

    if uploaded_file is not None:
        string_data = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(io.StringIO(string_data), header=None, names=['Ticker'])
        
        if df.empty:
            st.error("The uploaded CSV file is empty. Please check the content.")
        else:
            ticker_list = df['Ticker'].astype(str).str.strip().str.upper().tolist()
            valid_tickers = [t for t in ticker_list if len(t) > 0 and not t.isdigit()]
            
            if not valid_tickers:
                st.error("The file contains no valid stock tickers. Please ensure the column has tickers.")
            else:
                st.success(f"✅ Successfully loaded {len(valid_tickers)} tickers. Running analysis...")
                
                final_report_df = run_analysis(valid_tickers)
                
                st.header("📈 Analysis Results (Ranked by Score)")
                st.caption("Use the download button below to get the raw data for your Master Template.")
                
                st.dataframe(final_report_df, use_container_width=True)
                
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(final_report_df)

                st.download_button(
                    label="⬇️ Download Raw Data CSV",
                    data=csv,
                    file_name=f'Weekly_Analysis_Report_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                )
