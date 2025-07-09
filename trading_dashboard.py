import streamlit as st
import pandas as pd

def render_trading_dashboard():
    """Render the trading dashboard with options analysis"""
    
    st.markdown("## 📊 Trading Dashboard")
    st.markdown("Upload your trading data CSV from Merrill Lynch to view your portfolio.")
    
    # Strategy context
    with st.expander("📋 Strategy Overview", expanded=True):
        st.markdown("""
        **Current Strategy:**
        - **Covered Calls**: Selling call options against owned stock positions to generate premium income
        - **Cash-Secured Puts**: Selling put options with cash backing to potentially acquire stock at lower prices
        
        **Strategy Goals:**
        - Generate consistent income through option premiums
        - Reduce cost basis on stock positions
        - Potentially acquire stocks at attractive prices
        """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Trading Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load the CSV
            df = pd.read_csv(uploaded_file)
            
            # Display basic info
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            # Clean and process the data
            # Convert date column
            if 'COB Date' in df.columns:
                df['COB Date'] = pd.to_datetime(df['COB Date'])
            
            # Convert numeric columns, handling parentheses (negative values)
            numeric_columns = ['Quantity', 'Price ($)', 'Value ($)', 'Unrealized Gain/Loss ($)', 'Unrealized Gain/Loss (%)']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('(', '-').str.replace(')', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Identify options vs stocks
            df['Asset Type'] = 'Stock'
            df.loc[df['Symbol'].str.contains('#', na=False), 'Asset Type'] = 'Option'
            df.loc[df['Security Description'].str.contains('CALL|PUT', na=False), 'Asset Type'] = 'Option'
            
            # Extract options information
            if 'Security Description' in df.columns:
                # Extract option type (CALL/PUT)
                df['Option Type'] = df['Security Description'].str.extract(r'(CALL|PUT)')
                
                # Extract underlying symbol
                df['Underlying'] = df['Symbol'].str.extract(r'([A-Z]+)#')
                
                # Extract strike price
                df['Strike Price'] = df['Security Description'].str.extract(r'(\d+\.?\d*) EXP')
                
                # Extract expiration date
                df['Expiration'] = df['Security Description'].str.extract(r'EXP (\d{2}-\d{2}-\d{2})')
                
                # Determine strategy type based on option type and underlying ownership
                df['Strategy'] = 'Unknown'
                
                # Check for covered calls (CALL options with negative quantity = sold calls)
                covered_call_mask = (df['Option Type'] == 'CALL') & (df['Quantity'] < 0)
                df.loc[covered_call_mask, 'Strategy'] = 'Covered Call'
                
                # Check for cash-secured puts (PUT options with negative quantity = sold puts)
                cash_secured_put_mask = (df['Option Type'] == 'PUT') & (df['Quantity'] < 0)
                df.loc[cash_secured_put_mask, 'Strategy'] = 'Cash-Secured Put'
                
                # Check for long calls (CALL options with positive quantity)
                long_call_mask = (df['Option Type'] == 'CALL') & (df['Quantity'] > 0)
                df.loc[long_call_mask, 'Strategy'] = 'Long Call'
                
                # Check for long puts (PUT options with positive quantity)
                long_put_mask = (df['Option Type'] == 'PUT') & (df['Quantity'] > 0)
                df.loc[long_put_mask, 'Strategy'] = 'Long Put'
            
            # Show the data
            st.subheader("📈 Portfolio Overview")
            
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = df['Value ($)'].sum()
                st.metric("Total Portfolio Value", f"${total_value:,.2f}")
            
            with col2:
                total_pnl = df['Unrealized Gain/Loss ($)'].sum()
                pnl_color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}", delta_color=pnl_color)
            
            with col3:
                options_count = len(df[df['Asset Type'] == 'Option'])
                st.metric("Options Positions", options_count)
            
            with col4:
                stocks_count = len(df[df['Asset Type'] == 'Stock'])
                st.metric("Stock Positions", stocks_count)
            
            # Strategy Analysis
            if len(df[df['Asset Type'] == 'Option']) > 0:
                st.subheader("🎯 Options Strategy Analysis")
                
                options_df = df[df['Asset Type'] == 'Option'].copy()
                
                # Strategy breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    # Strategy distribution
                    if 'Strategy' in options_df.columns:
                        strategy_counts = options_df['Strategy'].value_counts()
                        st.write("**Strategy Distribution:**")
                        for strategy, count in strategy_counts.items():
                            st.write(f"- {strategy}: {count}")
                    
                    # Options by type
                    if 'Option Type' in options_df.columns:
                        option_types = options_df['Option Type'].value_counts()
                        st.write("**Options by Type:**")
                        for opt_type, count in option_types.items():
                            st.write(f"- {opt_type}: {count}")
                
                with col2:
                    # Strategy P&L
                    if 'Strategy' in options_df.columns:
                        strategy_pnl = options_df.groupby('Strategy')['Unrealized Gain/Loss ($)'].sum()
                        st.write("**P&L by Strategy:**")
                        for strategy, pnl in strategy_pnl.items():
                            color = "🟢" if pnl >= 0 else "🔴"
                            st.write(f"- {strategy}: {color} ${pnl:,.2f}")
                    
                    # Options value
                    options_value = options_df['Value ($)'].sum()
                    st.metric("Total Options Value", f"${options_value:,.2f}")
                
                # Covered Calls Analysis
                if len(options_df[options_df['Strategy'] == 'Covered Call']) > 0:
                    st.subheader("📞 Covered Calls Analysis")
                    
                    covered_calls = options_df[options_df['Strategy'] == 'Covered Call'].copy()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        cc_count = len(covered_calls)
                        st.metric("Covered Call Positions", cc_count)
                    
                    with col2:
                        cc_premium = covered_calls['Value ($)'].sum()
                        st.metric("Total Premium Collected", f"${cc_premium:,.2f}")
                    
                    with col3:
                        cc_pnl = covered_calls['Unrealized Gain/Loss ($)'].sum()
                        cc_pnl_color = "normal" if cc_pnl >= 0 else "inverse"
                        st.metric("Covered Call P&L", f"${cc_pnl:,.2f}", delta=f"{cc_pnl:,.2f}", delta_color=cc_pnl_color)
                    
                    # Covered call details
                    if len(covered_calls) > 0:
                        st.write("**Covered Call Details:**")
                        cc_display = covered_calls[['Underlying', 'Strike Price', 'Expiration', 'Quantity', 'Value ($)', 'Unrealized Gain/Loss ($)']].copy()
                        cc_display['Quantity'] = cc_display['Quantity'].abs()  # Show as positive for display
                        st.dataframe(cc_display, use_container_width=True)
                
                # Cash-Secured Puts Analysis
                if len(options_df[options_df['Strategy'] == 'Cash-Secured Put']) > 0:
                    st.subheader("💰 Cash-Secured Puts Analysis")
                    
                    cash_puts = options_df[options_df['Strategy'] == 'Cash-Secured Put'].copy()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csp_count = len(cash_puts)
                        st.metric("Cash-Secured Put Positions", csp_count)
                    
                    with col2:
                        csp_premium = cash_puts['Value ($)'].sum()
                        st.metric("Total Premium Collected", f"${csp_premium:,.2f}")
                    
                    with col3:
                        csp_pnl = cash_puts['Unrealized Gain/Loss ($)'].sum()
                        csp_pnl_color = "normal" if csp_pnl >= 0 else "inverse"
                        st.metric("Cash-Secured Put P&L", f"${csp_pnl:,.2f}", delta=f"{csp_pnl:,.2f}", delta_color=csp_pnl_color)
                    
                    # Cash-secured put details
                    if len(cash_puts) > 0:
                        st.write("**Cash-Secured Put Details:**")
                        csp_display = cash_puts[['Underlying', 'Strike Price', 'Expiration', 'Quantity', 'Value ($)', 'Unrealized Gain/Loss ($)']].copy()
                        csp_display['Quantity'] = csp_display['Quantity'].abs()  # Show as positive for display
                        st.dataframe(csp_display, use_container_width=True)
            
            # Stock Analysis with Strategy Context
            if len(df[df['Asset Type'] == 'Stock']) > 0:
                st.subheader("📈 Stock Analysis")
                
                stocks_df = df[df['Asset Type'] == 'Stock'].copy()
                
                # Check which stocks have covered calls
                if 'Underlying' in df.columns:
                    stocks_with_cc = df[df['Strategy'] == 'Covered Call']['Underlying'].unique()
                    stocks_df['Has Covered Call'] = stocks_df['Symbol'].isin(stocks_with_cc)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top performing stocks
                    top_performers = stocks_df.nlargest(3, 'Unrealized Gain/Loss ($)')
                    st.write("**Top Performers:**")
                    for _, row in top_performers.iterrows():
                        symbol = row['Symbol']
                        pnl = row['Unrealized Gain/Loss ($)']
                        pnl_pct = row['Unrealized Gain/Loss (%)']
                        cc_status = " (CC)" if row.get('Has Covered Call', False) else ""
                        st.write(f"- {symbol}{cc_status}: ${pnl:,.2f} ({pnl_pct:.1f}%)")
                
                with col2:
                    # Worst performing stocks
                    worst_performers = stocks_df.nsmallest(3, 'Unrealized Gain/Loss ($)')
                    st.write("**Worst Performers:**")
                    for _, row in worst_performers.iterrows():
                        symbol = row['Symbol']
                        pnl = row['Unrealized Gain/Loss ($)']
                        pnl_pct = row['Unrealized Gain/Loss (%)']
                        cc_status = " (CC)" if row.get('Has Covered Call', False) else ""
                        st.write(f"- {symbol}{cc_status}: ${pnl:,.2f} ({pnl_pct:.1f}%)")
            
            # Strategy Performance Summary
            if len(df[df['Asset Type'] == 'Option']) > 0:
                st.subheader("📊 Strategy Performance Summary")
                
                # Calculate strategy metrics
                strategy_summary = []
                
                # Covered Calls
                if len(options_df[options_df['Strategy'] == 'Covered Call']) > 0:
                    cc_data = options_df[options_df['Strategy'] == 'Covered Call']
                    cc_premium = cc_data['Value ($)'].sum()
                    cc_pnl = cc_data['Unrealized Gain/Loss ($)'].sum()
                    cc_roi = (cc_pnl / cc_premium * 100) if cc_premium != 0 else 0
                    strategy_summary.append({
                        'Strategy': 'Covered Calls',
                        'Positions': len(cc_data),
                        'Premium Collected': cc_premium,
                        'Current P&L': cc_pnl,
                        'ROI %': cc_roi
                    })
                
                # Cash-Secured Puts
                if len(options_df[options_df['Strategy'] == 'Cash-Secured Put']) > 0:
                    csp_data = options_df[options_df['Strategy'] == 'Cash-Secured Put']
                    csp_premium = csp_data['Value ($)'].sum()
                    csp_pnl = csp_data['Unrealized Gain/Loss ($)'].sum()
                    csp_roi = (csp_pnl / csp_premium * 100) if csp_premium != 0 else 0
                    strategy_summary.append({
                        'Strategy': 'Cash-Secured Puts',
                        'Positions': len(csp_data),
                        'Premium Collected': csp_premium,
                        'Current P&L': csp_pnl,
                        'ROI %': csp_roi
                    })
                
                if strategy_summary:
                    summary_df = pd.DataFrame(strategy_summary)
                    st.dataframe(summary_df, use_container_width=True)
            
            # Detailed data view
            st.subheader("📋 Detailed Portfolio Data")
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            with col1:
                asset_filter = st.selectbox("Filter by Asset Type", ["All", "Options", "Stocks"])
            
            with col2:
                if 'Option Type' in df.columns:
                    option_filter = st.selectbox("Filter by Option Type", ["All", "CALL", "PUT"])
            
            with col3:
                if 'Strategy' in df.columns:
                    strategy_filter = st.selectbox("Filter by Strategy", ["All", "Covered Call", "Cash-Secured Put", "Long Call", "Long Put"])
            
            # Apply filters
            filtered_df = df.copy()
            if asset_filter == "Options":
                filtered_df = filtered_df[filtered_df['Asset Type'] == 'Option']
            elif asset_filter == "Stocks":
                filtered_df = filtered_df[filtered_df['Asset Type'] == 'Stock']
            
            if 'Option Type' in filtered_df.columns and option_filter != "All":
                filtered_df = filtered_df[filtered_df['Option Type'] == option_filter]
            
            if 'Strategy' in filtered_df.columns and strategy_filter != "All":
                filtered_df = filtered_df[filtered_df['Strategy'] == strategy_filter]
            
            # Display filtered data
            st.dataframe(filtered_df, use_container_width=True)
            
            # Show column info
            st.subheader("📋 Data Structure")
            st.write("Available columns:", list(df.columns))
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
            st.info("💡 Make sure your CSV file is properly formatted.")
    else:
        st.info("📁 Please upload a CSV file with your trading data from Merrill Lynch.")
        st.markdown("""
        **Expected CSV format:**
        - COB Date, Symbol, Security Description, Quantity, Price ($), Value ($), etc.
        - You can export this from Merrill Edge account
        """) 