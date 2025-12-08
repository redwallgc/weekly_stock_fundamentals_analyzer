# app.py - Streamlit Application Code
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import io
# NOTE: xlsxwriter is not needed here as we won't be writing formatted Excel sheets in Streamlit

# Configuration
st.set_page_config(
    page_title="Weekly Stock Analysis Tool",
    layout="wide"
)

# Application Title
st.title("📊 Weekly Fundamental Analysis Tool")
st.markdown("Upload your single-column CSV of stock tickers below to run the analysis.")

# ----------------------------------------------------------------------
# 📌 1. DEFINE ANALYSIS LOGIC AND UI
# ----------------------------------------------------------------------

# Define the Column Order and Scoring Weights (Copied from Colab Step 2)
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

# Define the analysis function (Copied from Colab Step 2)
# We use st.cache_data to speed up the app by caching data fetching
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
            # Code from your Colab Step 2 (Analysis) - START
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
            
            # --- A. Balance Sheet (Current Ratio) ---
            current_assets = safe_get(latest_bs, 'Current Assets')
            current_liabilities = safe_get(latest_bs, 'Current Liabilities')
            current_ratio = current_assets / current_liabilities if current_liabilities and np.isfinite(current_liabilities) and current_liabilities != 0 else np.nan
            cr_pass = np.isfinite(current_ratio) and current_ratio >= 1.0
            
            # --- B. Income Statement (Operating Margin) ---
            operating_income = safe_get(latest_is, 'Operating Income')
            total_revenue = safe_get(latest_is, 'Total Revenue')
            operating_margin = (operating_income / total_revenue) * 100 if total_revenue and np.isfinite(total_revenue) and total_revenue != 0 else np.nan
            om_pass = np.isfinite(operating_margin) and operating_margin >= 15.0

            # --- C. Valuation Ratios (P/E and P/S) ---
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

            # --------------------------------------
            # 4. CALCULATE SCORE AND MAP VALUES TO FINAL COLUMN NAMES
            # --------------------------------------
            
            pass_fail_statuses = {
                'CR Pass? (>=1)': cr_pass,
                'OM Pass? (>=15)': om_pass,
                'P/E Pass? (>=20)': pe_pass,
                'P/S Pass? (<=2)': ps_pass,
                'FCF Pass?': fcf_pass,
                'Target Pass? (P<T)': target_pass
            }

            # Calculate Score
            for metric, is_passing in pass_fail_statuses.items():
                if is_passing:
                    score += SCORING_WEIGHTS.get(metric, 0)
            
            data['Overall Score'] = score

            # Map results to final columns
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
            # Code from your Colab Step 2 (Analysis) - END
                
        except Exception:
            # print(f"An error occurred fetching data for {ticker_symbol}: {e}") # Suppress error printing in web app
            pass
        
        analysis_results.append(data)

    # Execute the analysis
    for ticker in ticker_list:
        analyze_ticker(ticker)

    # Create and sort the final DataFrame
    final_df = pd.DataFrame(analysis_results)
    final_df = final_df.sort_values(by='Overall Score', ascending=False)
    final_df = final_df[COLUMN_ORDER]
    
    return final_df

# ----------------------------------------------------------------------
# 📌 2. STREAMLIT INPUT HANDLING
# ----------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "1. Choose a CSV file (single column of tickers)", 
    type=["csv"]
)

if uploaded_file is not None:
    # Read the uploaded file into a string
    string_data = uploaded_file.getvalue().decode('utf-8')
    
    # --- FIX APPLIED HERE ---
    # Force pandas to read the file with NO HEADER (header=None) so all rows are data.
    # We then rename the single column to 'Ticker'.
    df = pd.read_csv(io.StringIO(string_data), header=None, names=['Ticker'])
    # --- END FIX ---
    
    if df.empty:
        st.error("The uploaded CSV file is empty. Please check the content.")
    else:
        # Extract, strip, and uppercase the tickers from the first column (now correctly named 'Ticker')
        ticker_list = df['Ticker'].astype(str).str.strip().str.upper().tolist()
        
        # Filter out any non-ticker values or blanks
        valid_tickers = [t for t in ticker_list if len(t) > 0 and not t.isdigit()]
        
        if not valid_tickers:
            st.error("The file contains no valid stock tickers. Please ensure the column has tickers.")
        else:
            st.success(f"✅ Successfully loaded {len(valid_tickers)} tickers. Running analysis...")
            
            # Run the analysis function
            final_report_df = run_analysis(valid_tickers)
            
            st.header("📈 Analysis Results (Ranked by Score)")
            st.caption("Use the download button below to get the raw data for your Master Template.")
            
            # Display the DataFrame in Streamlit
            st.dataframe(final_report_df, use_container_width=True)
            
            # --- Download Button (Replaces downloading from Colab's file panel) ---
            
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
