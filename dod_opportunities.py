import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime, timedelta
import json
import os

def load_prompt(filename):
    """Load a prompt from the prompts directory"""
    prompt_path = os.path.join("prompts", filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"❌ Prompt file not found: {prompt_path}")
        return ""

def scrape_sam_gov_opportunities():
    """Scrape SAM.gov for AI-related opportunities using the official API"""
    
    opportunities = []
    
    # Get API key from Streamlit secrets
    try:
        api_key = st.secrets["sam_gov"]["api_key"]
    except KeyError:
        st.error("❌ SAM.gov API key not found in Streamlit secrets.")
        st.info("💡 Please add your API key to `.streamlit/secrets.toml` file.")
        return []
    
    try:
        # Use the official SAM.gov API
        api_url = "https://api.sam.gov/opportunities/v2/search"
        
        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Format dates as MM/dd/yyyy
        posted_from = start_date.strftime("%m/%d/%Y")
        posted_to = end_date.strftime("%m/%d/%Y")
        
        # Search parameters for AI opportunities
        params = {
            'api_key': api_key,
            'postedFrom': posted_from,
            'postedTo': posted_to,
            'limit': 100,  # Get up to 100 opportunities
            'offset': 0,
            'title': 'artificial intelligence'  # Search for AI in title
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json'
        }
        
        # Make API request
        response = requests.get(api_url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'opportunitiesData' in data and data['opportunitiesData']:
                for item in data['opportunitiesData']:
                    # Check if this is an AI-related opportunity
                    title = item.get('title', '').lower()
                    description = item.get('description', '').lower()
                    
                    # AI-related keywords to look for
                    ai_keywords = [
                        'artificial intelligence', 'ai', 'machine learning', 'ml',
                        'large language model', 'llm', 'generative ai', 'deep learning',
                        'neural network', 'natural language processing', 'nlp',
                        'computer vision', 'chatbot', 'gpt', 'openai', 'claude'
                    ]
                    
                    # Check if any AI keywords are in title or description
                    is_ai_related = any(keyword in title or keyword in description for keyword in ai_keywords)
                    
                    if is_ai_related:
                        opportunities.append({
                            'Title': item.get('title', 'No title'),
                            'Agency': item.get('fullParentPathName', 'Unknown Agency'),
                            'Description': item.get('description', 'No description'),
                            'Date': item.get('postedDate', 'Unknown Date'),
                            'ID': item.get('noticeId', 'No ID'),
                            'Link': item.get('uiLink', ''),
                            'Full_Text': f"{item.get('title', '')} {item.get('description', '')}",
                            'Type': item.get('type', 'Unknown'),
                            'SolicitationNumber': item.get('solicitationNumber', ''),
                            'ResponseDeadline': item.get('responseDeadLine', ''),
                            'Active': item.get('active', 'Unknown')
                        })
            
            # If no AI opportunities found in title search, try broader search
            if not opportunities:
                st.info("No AI opportunities found in titles. Trying broader search...")
                
                # Search without title filter to get more opportunities
                params.pop('title', None)
                response = requests.get(api_url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'opportunitiesData' in data and data['opportunitiesData']:
                        for item in data['opportunitiesData']:
                            title = item.get('title', '').lower()
                            description = item.get('description', '').lower()
                            
                            # Check if any AI keywords are in title or description
                            is_ai_related = any(keyword in title or keyword in description for keyword in ai_keywords)
                            
                            if is_ai_related:
                                opportunities.append({
                                    'Title': item.get('title', 'No title'),
                                    'Agency': item.get('fullParentPathName', 'Unknown Agency'),
                                    'Description': item.get('description', 'No description'),
                                    'Date': item.get('postedDate', 'Unknown Date'),
                                    'ID': item.get('noticeId', 'No ID'),
                                    'Link': item.get('uiLink', ''),
                                    'Full_Text': f"{item.get('title', '')} {item.get('description', '')}",
                                    'Type': item.get('type', 'Unknown'),
                                    'SolicitationNumber': item.get('solicitationNumber', ''),
                                    'ResponseDeadline': item.get('responseDeadLine', ''),
                                    'Active': item.get('active', 'Unknown')
                                })
            
            # Limit to first 25 opportunities for display
            opportunities = opportunities[:25]
            
        elif response.status_code == 401:
            st.error("❌ API key authentication failed. Please check your SAM.gov API key in the secrets file.")
            return []
        elif response.status_code == 403:
            st.error("❌ Access forbidden. Your API key may not have the required permissions.")
            return []
        elif response.status_code == 404:
            st.warning("⚠️ No opportunities found for the specified criteria.")
            return []
        else:
            st.error(f"❌ API request failed with status code: {response.status_code}")
            st.info(f"Response: {response.text[:200]}...")
            return []
            
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. SAM.gov API may be experiencing issues.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return []
    
    return opportunities

def analyze_ai_relevance(opportunities, llm):
    """Analyze opportunities for AI/LLM/Generative AI relevance"""
    
    if not opportunities:
        return []
    
    analyzed_opportunities = []
    
    # Load the AI analysis prompt
    prompt_template = load_prompt("dod_ai_analysis_prompt.txt")
    
    for opp in opportunities:
        try:
            if prompt_template:
                # Create analysis prompt using template
                analysis_prompt = prompt_template.format(
                    title=opp['Title'],
                    description=opp['Description']
                )
            else:
                # Fallback prompt if template not found
                analysis_prompt = f"""
                Analyze this government contract opportunity for AI/LLM/Generative AI relevance:
                
                TITLE: {opp['Title']}
                DESCRIPTION: {opp['Description']}
                
                Determine if this opportunity specifically requests or mentions:
                1. Artificial Intelligence (AI)
                2. Large Language Models (LLMs)
                3. Generative AI tools
                4. Machine Learning
                5. Natural Language Processing
                6. AI/ML development or implementation
                
                Respond with a JSON object:
                {{
                    "is_ai_related": true/false,
                    "ai_technologies": ["list", "of", "specific", "ai", "technologies"],
                    "relevance_score": 0-100,
                    "summary": "brief explanation of AI relevance",
                    "recommendation": "apply/consider/pass"
                }}
                
                Only mark as AI-related if there are explicit mentions or clear requirements for AI technologies.
                """
            
            # Get AI analysis
            try:
                analysis_response = llm.predict(analysis_prompt)
                
                # Try to parse JSON response
                try:
                    analysis = json.loads(analysis_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, create a basic analysis
                    analysis = {
                        "is_ai_related": "artificial intelligence" in opp['Full_Text'].lower() or 
                                       "llm" in opp['Full_Text'].lower() or 
                                       "generative ai" in opp['Full_Text'].lower(),
                        "ai_technologies": [],
                        "relevance_score": 50,
                        "summary": "Basic keyword analysis",
                        "recommendation": "consider"
                    }
                
            except Exception as e:
                # Fallback analysis using keyword matching
                text_lower = opp['Full_Text'].lower()
                ai_keywords = ['artificial intelligence', 'ai', 'llm', 'large language model', 
                             'generative ai', 'machine learning', 'ml', 'nlp', 'natural language processing',
                             'deep learning', 'neural network', 'chatbot', 'gpt', 'openai', 'claude']
                
                ai_mentions = [keyword for keyword in ai_keywords if keyword in text_lower]
                
                analysis = {
                    "is_ai_related": len(ai_mentions) > 0,
                    "ai_technologies": ai_mentions,
                    "relevance_score": min(100, len(ai_mentions) * 20),
                    "summary": f"Found {len(ai_mentions)} AI-related keywords: {', '.join(ai_mentions)}",
                    "recommendation": "apply" if len(ai_mentions) >= 2 else "consider" if len(ai_mentions) >= 1 else "pass"
                }
            
            # Add analysis to opportunity
            opp.update(analysis)
            analyzed_opportunities.append(opp)
            
        except Exception as e:
            st.warning(f"Error analyzing opportunity: {str(e)}")
            opp.update({
                "is_ai_related": False,
                "ai_technologies": [],
                "relevance_score": 0,
                "summary": "Analysis failed",
                "recommendation": "pass"
            })
            analyzed_opportunities.append(opp)
    
    return analyzed_opportunities

def render_dod_opportunities(llm):
    """Render the DoD opportunities dashboard"""
    
    st.markdown("## 🎯 DoD AI Opportunities")
    st.markdown("Searching for government contracts related to Artificial Intelligence, LLMs, and Generative AI tools.")
    
    # Strategy context
    with st.expander("📋 About DoD AI Opportunities", expanded=True):
        st.markdown("""
        **What we're looking for:**
        - **Artificial Intelligence (AI)** development and implementation
        - **Large Language Models (LLMs)** integration and deployment
        - **Generative AI** tools and applications
        - **Machine Learning** solutions
        - **Natural Language Processing** capabilities
        
        **Source:** [SAM.gov](https://sam.gov) - Official government contracting portal
        
        **Note:** Due to SAM.gov's modern web architecture, we may use sample data for demonstration.
        """)
    
    # Search controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_keywords = st.text_input(
            "Additional Keywords (optional)", 
            placeholder="e.g., machine learning, NLP, chatbot",
            help="Add specific AI technologies to search for"
        )
    
    with col2:
        search_button = st.button("🔍 Search Opportunities", type="primary")
    
    # Main content area
    if search_button:
        with st.spinner("Searching SAM.gov for AI opportunities..."):
            
            # Scrape opportunities
            opportunities = scrape_sam_gov_opportunities()
            
            if opportunities:
                st.success(f"✅ Found {len(opportunities)} opportunities")
                
                # Analyze for AI relevance
                with st.spinner("Analyzing opportunities for AI relevance..."):
                    analyzed_opportunities = analyze_ai_relevance(opportunities, llm)
                
                # Filter for AI-related opportunities
                ai_opportunities = [opp for opp in analyzed_opportunities if opp.get('is_ai_related', False)]
                
                if ai_opportunities:
                    st.success(f"🎯 Found {len(ai_opportunities)} AI-related opportunities!")
                    
                    # Sort by relevance score
                    ai_opportunities.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    
                    # Display AI opportunities in the main column
                    st.subheader("🤖 AI-Related Opportunities")
                    
                    for i, opp in enumerate(ai_opportunities):
                        with st.expander(f"📋 {opp['Title'][:100]}...", expanded=(i < 3)):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Agency:** {opp['Agency']}")
                                st.markdown(f"**ID:** {opp['ID']}")
                                st.markdown(f"**Date:** {opp['Date']}")
                                
                                if opp['Description']:
                                    st.markdown("**Description:**")
                                    st.write(opp['Description'][:300] + "..." if len(opp['Description']) > 300 else opp['Description'])
                                
                                st.markdown(f"**AI Technologies:** {', '.join(opp.get('ai_technologies', []))}")
                                st.markdown(f"**Summary:** {opp.get('summary', 'No summary available')}")
                            
                            with col2:
                                relevance_score = opp.get('relevance_score', 0)
                                st.metric("Relevance Score", f"{relevance_score}/100")
                                
                                recommendation = opp.get('recommendation', 'pass')
                                if recommendation == 'apply':
                                    st.success("✅ Apply")
                                elif recommendation == 'consider':
                                    st.warning("🤔 Consider")
                                else:
                                    st.info("⏭️ Pass")
                                
                                if opp['Link']:
                                    st.link_button("🔗 View Details", opp['Link'])
                    
                    # Summary statistics
                    st.subheader("📊 AI Opportunities Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_score = sum(opp.get('relevance_score', 0) for opp in ai_opportunities) / len(ai_opportunities)
                        st.metric("Avg Relevance Score", f"{avg_score:.1f}/100")
                    
                    with col2:
                        apply_count = len([opp for opp in ai_opportunities if opp.get('recommendation') == 'apply'])
                        st.metric("Apply", apply_count)
                    
                    with col3:
                        consider_count = len([opp for opp in ai_opportunities if opp.get('recommendation') == 'consider'])
                        st.metric("Consider", consider_count)
                    
                    with col4:
                        pass_count = len([opp for opp in ai_opportunities if opp.get('recommendation') == 'pass'])
                        st.metric("Pass", pass_count)
                    
                    # Agency breakdown
                    st.subheader("🏛️ Opportunities by Agency")
                    agency_counts = {}
                    for opp in ai_opportunities:
                        agency = opp['Agency']
                        agency_counts[agency] = agency_counts.get(agency, 0) + 1
                    
                    for agency, count in sorted(agency_counts.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- **{agency}**: {count} opportunities")
                    
                    # Export data
                    st.subheader("📥 Export Data")
                    if st.button("📊 Export to CSV"):
                        df = pd.DataFrame(ai_opportunities)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv,
                            file_name=f"dod_ai_opportunities_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.warning("❌ No AI-related opportunities found in the current search results.")
                    st.info("💡 Try adjusting your search keywords or check back later for new opportunities.")
                    
                    # Show all opportunities for reference
                    st.subheader("📋 All Opportunities Found")
                    for opp in analyzed_opportunities[:5]:  # Show first 5
                        with st.expander(f"📋 {opp['Title'][:100]}..."):
                            st.write(f"**Agency:** {opp['Agency']}")
                            st.write(f"**Description:** {opp['Description'][:200]}...")
                            st.write(f"**AI Relevance:** {opp.get('summary', 'Not analyzed')}")
            
            else:
                st.error("❌ No opportunities found. The website structure may have changed or there might be a connection issue.")
                st.info("💡 You can manually visit [SAM.gov](https://sam.gov/search/?index=opp&page=1&pageSize=25&sort=-modifiedDate&sfm%5Bstatus%5D%5Bis_active%5D=true&sfm%5BsimpleSearch%5D%5BkeywordRadio%5D=ALL&sfm%5BsimpleSearch%5D%5BkeywordTags%5D%5B0%5D%5Bvalue%5D=artificial%20intelligence) to search manually.")
    
    # Manual search option
    st.subheader("🔗 Manual Search")
    st.markdown("""
    If the automated search doesn't work, you can manually search SAM.gov:
    
    **Direct Link:** [SAM.gov AI Opportunities](https://sam.gov/search/?index=opp&page=1&pageSize=25&sort=-modifiedDate&sfm%5Bstatus%5D%5Bis_active%5D=true&sfm%5BsimpleSearch%5D%5BkeywordRadio%5D=ALL&sfm%5BsimpleSearch%5D%5BkeywordTags%5D%5B0%5D%5Bvalue%5D=artificial%20intelligence)
    
    **Search Tips:**
    - Use keywords: "artificial intelligence", "AI", "LLM", "generative AI", "machine learning"
    - Filter by active opportunities
    - Sort by most recent
    - Look for opportunities from DoD, DARPA, and other defense agencies
    """)
    
    # Disclaimer
    st.info("""
    **Disclaimer:** This tool attempts to scrape publicly available data from SAM.gov. 
    Due to the site's modern architecture, we may use sample data for demonstration.
    Always verify opportunity details on the official SAM.gov website before applying.
    """) 