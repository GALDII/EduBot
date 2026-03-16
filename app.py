import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import uuid

from utils.document_loader import parse_pdf
from utils.vector_store import add_chunks, clear_collection, get_stored_sources
from utils.rag_chain import ask
from utils.voice import text_to_speech, speech_to_text
from utils.data_analyzer import load_csv, query_csv, generate_chart, generate_dashboard_analysis
from utils.cache import get_cached_response, cache_response, clear_cache
from utils.validation import validate_file, validate_csv_structure
from utils.database import save_conversation, load_conversations, save_user_profile, load_user_profiles, save_bookmark, load_bookmarks
from utils.career_analysis import calculate_career_readiness_score, analyze_skill_gaps, analyze_performance_trends, get_recommendations, compare_with_benchmarks
from utils.visualizations import create_interactive_chart, create_performance_heatmap, create_comparison_chart, create_timeline_chart, create_skill_gap_chart, create_readiness_radar, export_chart
from utils.language import translate_query, translate_response, detect_language
from utils.export import export_chat_markdown, export_chat_html, export_chat_pdf, export_chat_word
from utils.retry import retry_with_backoff

st.set_page_config(page_title="Analyser Bot", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main { 
        background: linear-gradient(180deg, #0a0d14 0%, #0f1117 100%);
    }
    .stChatMessage { 
        border-radius: 16px; 
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .source-badge {
        display: inline-block; 
        padding: 4px 12px; 
        border-radius: 12px;
        font-size: 11px; 
        font-weight: 600; 
        margin-bottom: 6px; 
        letter-spacing: 0.3px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .badge-docs  { background: linear-gradient(135deg, #1a3a2a, #0f2a1a); color: #4ade80; border: 1px solid #22c55e40; }
    .badge-web   { background: linear-gradient(135deg, #1a2a3a, #0f1a2a); color: #60a5fa; border: 1px solid #3b82f640; }
    .badge-both  { background: linear-gradient(135deg, #2a1a3a, #1a0f2a); color: #c084fc; border: 1px solid #a855f740; }
    .badge-llm   { background: linear-gradient(135deg, #2a2a1a, #1a1a0f); color: #facc15; border: 1px solid #eab30840; }
    .badge-sql   { background: linear-gradient(135deg, #3a2a1a, #2a1a0f); color: #f97316; border: 1px solid #f9731640; }
    .block-container { padding-bottom: 100px; }
    .stAudio { display: none; }
    .streamlit-mic-recorder { background: transparent !important; border: none !important; }
    iframe[title="streamlit_mic_recorder"] { 
        background: transparent !important; 
        min-height: 0 !important;
        height: 60px !important;
    }
    .intro-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
        border: 1px solid #2a3a5a; 
        border-radius: 20px;
        padding: 32px 36px; 
        margin-bottom: 28px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .feature-grid {
        display: grid; 
        grid-template-columns: repeat(3, 1fr);
        gap: 16px; 
        margin-top: 20px;
    }
    .feature-card {
        background: linear-gradient(135deg, #1a1f2e, #0f1419); 
        border: 1px solid #2a3a5a;
        border-radius: 14px; 
        padding: 18px 20px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border-color: #3a4a6a;
    }
    .feature-card h4 { 
        margin: 0 0 8px 0; 
        font-size: 15px; 
        color: #e2e8f0; 
        font-weight: 600;
    }
    .feature-card p  { 
        margin: 0; 
        font-size: 12.5px; 
        color: #94a3b8; 
        line-height: 1.5;
    }
    .query-chip {
        display: inline-block; 
        background: linear-gradient(135deg, #1e293b, #0f1419); 
        border: 1px solid #334155;
        border-radius: 20px; 
        padding: 6px 16px; 
        margin: 5px;
        font-size: 12px; 
        color: #cbd5e1; 
        cursor: default;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .query-chip:hover {
        background: linear-gradient(135deg, #2a3441, #1a1f2e);
        border-color: #475569;
        transform: scale(1.05);
    }
    .step-row { 
        display: flex; 
        gap: 16px; 
        margin-top: 16px; 
    }
    .step { 
        background: linear-gradient(135deg, #1a1f2e, #0f1419); 
        border-radius: 12px; 
        padding: 14px 18px; 
        flex: 1;
        font-size: 12.5px; 
        color: #94a3b8; 
        border: 1px solid #2a3a5a;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .step strong { 
        color: #60a5fa; 
        display: block; 
        margin-bottom: 6px; 
        font-size: 13px;
    }
    iframe[title="streamlit_mic_recorder.streamlit_mic_recorder"] {
        min-height: 0 !important;
    }
    div[data-testid="stIFrame"] > iframe {
        background: transparent !important;
    }
    /* Dashboard Enhancements */
    .dashboard-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid #2a3a5a;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1f2e, #0f1419);
        border: 1px solid #2a3a5a;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border-color: #3a4a6a;
    }
    .stat-value {
        font-size: 32px;
        font-weight: 700;
        color: #60a5fa;
        margin: 8px 0;
        text-shadow: 0 2px 4px rgba(96,165,250,0.3);
    }
    .stat-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .file-card {
        background: linear-gradient(135deg, #1a1f2e, #0f1419);
        border: 1px solid #2a3a5a;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .file-card:hover {
        border-color: #3a4a6a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .file-icon {
        font-size: 24px;
        margin-right: 12px;
        vertical-align: middle;
    }
    .file-name {
        font-size: 15px;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .file-meta {
        font-size: 12px;
        color: #94a3b8;
    }
    .summary-box {
        background: linear-gradient(135deg, #1a1f2e, #0f1419);
        border-left: 4px solid #60a5fa;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .chart-container {
        background: linear-gradient(135deg, #1a1f2e, #0f1419);
        border: 1px solid #2a3a5a;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #e2e8f0;
        margin: 24px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid #2a3a5a;
    }
    .divider-enhanced {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a3a5a, transparent);
        margin: 24px 0;
    }
    /* Button enhancements */
    .stButton > button {
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    /* Sidebar enhancements */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #0a0d14 100%);
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    /* File uploader enhancement */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #1a1f2e, #0f1419);
        border-radius: 12px;
        padding: 20px;
        border: 2px dashed #2a3a5a;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #3a4a6a;
    }
    /* Success/Info message styling */
    .stSuccess, .stInfo {
        border-radius: 12px;
        border-left: 4px solid;
    }
    .stSuccess {
        border-left-color: #4ade80;
    }
    .stInfo {
        border-left-color: #60a5fa;
    }
    /* Chat input enhancement */
    [data-testid="stChatInput"] {
        border-radius: 12px;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1a1f2e, #0f1419);
        border-radius: 8px;
        border: 1px solid #2a3a5a;
    }
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

BADGE = {
    "docs":    '<span class="source-badge badge-docs">📄 From Documents</span>',
    "web":     '<span class="source-badge badge-web">🌐 From Web Search</span>',
    "both":    '<span class="source-badge badge-both">📄🌐 Documents + Web</span>',
    "llm":     '<span class="source-badge badge-llm">🤖 General Knowledge</span>',
    "sql":     '<span class="source-badge badge-sql">📊 Data Analysis</span>',
    "sql+docs":'<span class="source-badge badge-both">📊📄 Data + Documents</span>',
    "sql+web": '<span class="source-badge badge-both">📊🌐 Data + Web</span>',
    "all":     '<span class="source-badge badge-both">📊📄🌐 All Sources</span>',
    "error":   '<span class="source-badge badge-llm">⚠️ Error</span>',
}


def handle_upload(uploaded_files):
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        return
    for f in new_files:
        # Validate file
        is_valid, error_msg = validate_file(f)
        if not is_valid:
            st.error(f"❌ {f.name}: {error_msg}")
            continue
        
        with st.spinner(f"Processing {f.name}..."):
            try:
                if f.name.endswith(".csv"):
                    df = load_csv(f)
                    # Validate CSV structure
                    is_valid_csv, csv_error = validate_csv_structure(df)
                    if not is_valid_csv:
                        st.error(f"❌ {f.name}: {csv_error}")
                        continue
                    st.session_state.csv_dataframes[f.name] = df
                    f.seek(0)
                    raw = f.read()
                    st.session_state.csv_bytes[f.name] = raw
                    import os
                    os.makedirs("chroma_db", exist_ok=True)
                    open(f"chroma_db/csv_{f.name}", "wb").write(raw)
                    st.success(f"✅ {f.name} processed successfully")
                else:
                    chunks = parse_pdf(f)
                    add_chunks(chunks)
                    full_text = " ".join(c["text"] for c in chunks)
                    st.session_state.pdf_texts[f.name] = full_text
                    st.success(f"✅ {f.name} processed successfully")
                st.session_state.processed_files.add(f.name)
            except Exception as e:
                st.error(f"❌ Failed to process {f.name}: {e}")
    st.rerun()


def render_sidebar():
    try:
        audio = None
        mode = "detailed"
        with st.sidebar:
            st.markdown("""
            <div style="text-align:center; padding:20px 0; border-bottom:2px solid #2a3a5a; margin-bottom:20px;">
                <h1 style="margin:0; color:#e2e8f0; font-size:28px; font-weight:700;">🔬 Analyser Bot</h1>
                <p style="margin:8px 0 0 0; color:#94a3b8; font-size:13px;">Career Intelligence Assistant</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎙️ Voice Input")
            st.caption("Click to speak your question")
            audio = mic_recorder(start_prompt="🎙️ Speak", stop_prompt="⏹️ Stop", use_container_width=True, key="mic")
            
            st.divider()
            
            st.markdown("### ⚙️ Settings")
            mode = st.radio(
                "Response Mode", 
                ["Concise", "Detailed"], 
                index=1,
                help="Concise: Quick 2-3 sentence answers | Detailed: Full explanations"
            )
            
            st.divider()
            
            st.markdown("### 🛠️ Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All", use_container_width=True, help="Clear all files and chat history"):
                    for k in ["messages","processed_files","pdf_texts","csv_dataframes",
                              "csv_bytes","dashboard_analysed","dash_pdf_sums","dash_csv_sum","dash_chart_plan"]:
                        st.session_state[k] = [] if k == "messages" else ({} if k in ["pdf_texts","csv_dataframes","csv_bytes","dash_pdf_sums"] else (set() if k == "processed_files" else (False if k == "dashboard_analysed" else "")))
                    clear_collection()
                    import os, glob
                    for f in glob.glob("chroma_db/csv_*"):
                        os.remove(f)
                    st.success("✅ Cleared successfully!")
                    st.rerun()
            with col2:
                chat_text = "\n\n".join(
                    f"**{m['role'].capitalize()}:** {m['content']}"
                    for m in st.session_state.get("messages", [])
                )
                st.download_button(
                    "⬇️ Export Chat", 
                    data=chat_text or " ",
                    file_name="chat.md", 
                    mime="text/markdown",
                    use_container_width=True,
                    help="Download conversation as Markdown file"
                )
            
            # Quick Stats
            if st.session_state.get("messages"):
                st.divider()
                st.markdown("### 📊 Chat Stats")
                num_messages = len(st.session_state.messages)
                st.metric("Messages", num_messages)
        return mode.lower(), audio
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        return "detailed", None


def render_dashboard(combined_csv):
    """Tab 1 — Dashboard: upload, one Analyse All button, LLM summaries + smart charts."""
    try:
        # Header Section
        st.markdown("""
        <div class="dashboard-header">
            <h1 style="margin:0 0 8px 0; color:#e2e8f0; font-size:32px; font-weight:700;">📊 Dashboard</h1>
            <p style="margin:0; color:#94a3b8; font-size:15px;">Upload your files and get instant insights with AI-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # File Upload Section
        uploaded_files = st.file_uploader(
            "📁 Upload PDF(s) or CSV(s)", 
            type=["pdf", "csv"],
            accept_multiple_files=True, 
            key="uploader",
            help="Upload your resume PDFs, mark sheets CSV, or any relevant documents"
        )
        if uploaded_files:
            handle_upload(uploaded_files)

        stored = get_stored_sources()
        has_any = stored or st.session_state.get("csv_dataframes")
        
        if not has_any:
            st.markdown("""
            <div style="text-align:center; padding:40px; background:linear-gradient(135deg, #1a1f2e, #0f1419); 
                        border-radius:16px; border:2px dashed #2a3a5a; margin:20px 0;">
                <p style="font-size:48px; margin:0;">📂</p>
                <p style="color:#94a3b8; font-size:16px; margin-top:12px;">No files loaded yet. Upload PDFs or CSVs above to get started.</p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Stats Section
        stored = get_stored_sources()
        num_docs = len(stored) if stored else 0
        csv_name, df = None, None
        num_csvs = 0
        total_rows = 0
        total_cols = 0
        
        if st.session_state.get("csv_dataframes"):
            csv_name, df = next(iter(st.session_state.csv_dataframes.items()))
            num_csvs = len(st.session_state.csv_dataframes)
            total_rows = len(df) if df is not None else 0
            total_cols = len(df.columns) if df is not None else 0

        st.markdown("### 📈 Overview")
        stats_cols = st.columns(4)
        with stats_cols[0]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{num_docs}</div>
                <div class="stat-label">📄 Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with stats_cols[1]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{num_csvs}</div>
                <div class="stat-label">📊 CSV Files</div>
            </div>
            """, unsafe_allow_html=True)
        with stats_cols[2]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_rows:,}</div>
                <div class="stat-label">📋 Data Rows</div>
            </div>
            """, unsafe_allow_html=True)
        with stats_cols[3]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_cols}</div>
                <div class="stat-label">🔢 Columns</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider-enhanced"></div>', unsafe_allow_html=True)

        # Files Display Section
        st.markdown("### 📁 Uploaded Files")
        
        file_cols = st.columns(2)
        col_idx = 0
        
        # Display PDFs
        if stored:
            for src in stored:
                with file_cols[col_idx % 2]:
                    st.markdown(f"""
                    <div class="file-card">
                        <div class="file-name">
                            <span class="file-icon">📄</span>{src}
                        </div>
                        <div class="file-meta">PDF Document • Ready for analysis</div>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1
        
        # Display CSVs
        if st.session_state.get("csv_dataframes"):
            for name, csv_df in st.session_state.csv_dataframes.items():
                with file_cols[col_idx % 2]:
                    st.markdown(f"""
                    <div class="file-card">
                        <div class="file-name">
                            <span class="file-icon">📊</span>{name}
                        </div>
                        <div class="file-meta">{len(csv_df):,} rows × {len(csv_df.columns)} columns</div>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1

        st.markdown('<div class="divider-enhanced"></div>', unsafe_allow_html=True)

        # Analyse Button
        st.markdown("### 🔍 Analysis")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyse All Files", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your files... This may take a moment."):
                    try:
                        pdf_sums, csv_sum, chart_plan = generate_dashboard_analysis(
                            st.session_state.pdf_texts, csv_name, df
                        )
                        st.session_state["dash_pdf_sums"] = pdf_sums
                        st.session_state["dash_csv_sum"] = csv_sum
                        st.session_state["dash_chart_plan"] = chart_plan
                        st.session_state["dashboard_analysed"] = True
                        st.success("✅ Analysis complete! Scroll down to see insights.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")

        if not st.session_state.get("dashboard_analysed"):
            st.markdown("""
            <div style="text-align:center; padding:30px; background:linear-gradient(135deg, #1a1f2e, #0f1419); 
                        border-radius:14px; border:1px solid #2a3a5a; margin:20px 0;">
                <p style="color:#94a3b8; font-size:14px;">Click <strong style="color:#60a5fa;">Analyse All Files</strong> above to generate insights and visualizations</p>
            </div>
            """, unsafe_allow_html=True)
            return

        st.markdown('<div class="divider-enhanced"></div>', unsafe_allow_html=True)

        # ── PDF Summaries ─────────────────────────────────────────────────
        if stored:
            st.markdown("### 📄 Document Summaries")
            pdf_sums = st.session_state.get("dash_pdf_sums", {})
            for src in stored:
                summary = next(
                    (v for k, v in pdf_sums.items() if os.path.basename(k) == src or k == src),
                    "Summary unavailable."
                )
                st.markdown(f"""
                <div class="file-card">
                    <div class="file-name" style="margin-bottom:12px;">
                        <span class="file-icon">📄</span>{src}
                    </div>
                    <div class="summary-box">
                        <p style="margin:0; color:#cbd5e1; line-height:1.6; font-size:14px;">{summary}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── CSV Summary + Charts ──────────────────────────────────────────
        if df is not None:
            st.markdown("### 📊 Data Analysis & Visualizations")
            
            csv_sum = st.session_state.get("dash_csv_sum", "")
            if csv_sum:
                st.markdown(f"""
                <div class="summary-box">
                    <p style="margin:0; color:#cbd5e1; line-height:1.6; font-size:14px;">{csv_sum}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Data Preview
            with st.expander("📋 View Raw Data", expanded=False):
                st.dataframe(df, use_container_width=True, height=300)

            # Charts Section - Using Plotly
            chart_plan = st.session_state.get("dash_chart_plan", [])
            if not chart_plan:
                st.info("💡 No charts generated. Try asking questions in the Chat tab for custom visualizations.")
            else:
                st.markdown("### 📈 Generated Visualizations")
                # Display interactive Plotly charts
                for i in range(0, len(chart_plan), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(chart_plan):
                            chart = chart_plan[i + j]
                            with cols[j]:
                                try:
                                    x_col = chart.get("x")
                                    y_col = chart.get("y")
                                    chart_type = chart.get("type", "bar")
                                    title = chart.get("title", "Chart")
                                    
                                    if x_col and y_col and x_col in df.columns and y_col in df.columns:
                                        # Create interactive Plotly chart
                                        fig = create_interactive_chart(
                                            df, chart_type, x_col, y_col,
                                            title, chart.get("xlabel", x_col), chart.get("ylabel", y_col)
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Export options
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            png_data = export_chart(fig, "png")
                                            st.download_button("📥 PNG", png_data, f"{title}.png", "image/png", key=f"png_{i+j}")
                                        with col2:
                                            pdf_data = export_chart(fig, "pdf")
                                            st.download_button("📥 PDF", pdf_data, f"{title}.pdf", "application/pdf", key=f"pdf_{i+j}")
                                        with col3:
                                            svg_data = export_chart(fig, "svg")
                                            st.download_button("📥 SVG", svg_data, f"{title}.svg", "image/svg+xml", key=f"svg_{i+j}")
                                    else:
                                        # Fallback to matplotlib
                                        chart_bytes = generate_chart(df, "", chart)
                                        st.image(chart_bytes, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Chart generation error: {e}")
                
                # Additional visualizations
                st.markdown("### 📊 Additional Visualizations")
                
                # Performance Heatmap
                if len(df) > 0:
                    try:
                        fig_heatmap = create_performance_heatmap(df)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    except:
                        pass

    except Exception as e:
        st.error(f"Dashboard error: {e}")


def render_sql_result(sql_result: dict, msg_id: int = 0):
    """Render SQL query, result table, and visualize button below an assistant message."""
    if not sql_result or not sql_result.get("sql"):
        return
    with st.expander("🔍 View SQL Query", expanded=False):
        st.code(sql_result["sql"], language="sql")
    result_df = sql_result.get("result_df")
    if result_df is not None and not result_df.empty:
        st.markdown("**📊 Query Results:**")
        st.dataframe(result_df, use_container_width=True, height=min(300, len(result_df) * 35 + 50))
    if sql_result.get("can_visualize") and result_df is not None:
        if st.button("📈 Generate Visualization", key=f"chart_{msg_id}", use_container_width=True):
            with st.spinner("🎨 Creating chart..."):
                try:
                    chart_bytes = generate_chart(
                        result_df,
                        sql_result.get("sql", ""),
                        sql_result.get("chart_spec")
                    )
                    st.markdown("**📊 Visualization:**")
                    st.image(chart_bytes, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            source = msg.get("source", "llm")
            st.markdown(BADGE.get(source, BADGE["llm"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sql_result"):
                render_sql_result(msg["sql_result"], msg.get("id", 0))
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🔊 Listen", key=f"tts_{msg['id']}", use_container_width=True):
                    with st.spinner("🎵 Generating audio..."):
                        try:
                            audio = text_to_speech(msg["content"])
                            st.audio(audio, format="audio/mp3", autoplay=True)
                        except Exception as e:
                            st.error(f"TTS error: {e}")


def get_user_id() -> str:
    """Get or create user ID for session."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id


def render_career_analysis(combined_csv):
    """Render Career Analysis tab with readiness score, skill gaps, trends, etc."""
    try:
        st.markdown("### 🎯 Career Analysis")
        
        user_id = get_user_id()
        stored = get_stored_sources()
        has_resume = bool(stored)
        has_marks = combined_csv is not None and not combined_csv.empty
        
        if not has_resume and not has_marks:
            st.info("📝 Upload your resume and marks to get career analysis insights.")
            return
        
        # Target Role Input
        target_role = st.text_input("🎯 Target Role", value=st.session_state.get("target_role", ""), 
                                   placeholder="e.g., Data Engineer, Software Developer, Data Scientist")
        if target_role:
            st.session_state.target_role = target_role
        
        if not target_role:
            st.warning("Enter a target role to begin analysis.")
            return
        
        # Career Readiness Score
        if st.button("📊 Calculate Career Readiness Score", type="primary"):
            with st.spinner("Analyzing your profile..."):
                try:
                    resume_text = " ".join(st.session_state.pdf_texts.values()) if has_resume else ""
                    score_data = calculate_career_readiness_score(
                        resume_text, combined_csv, target_role
                    )
                    st.session_state.career_score = score_data
                except Exception as e:
                    st.error(f"Error calculating score: {e}")
        
        if "career_score" in st.session_state:
            score_data = st.session_state.career_score
            col1, col2 = st.columns([1, 2])
            
            with col1:
                overall = score_data.get("overall_score", 0)
                st.metric("Overall Score", f"{overall}/100")
                
                # Score breakdown
                st.markdown("#### Score Breakdown")
                st.metric("Academic", f"{score_data.get('academic_score', 0):.1f}/30")
                st.metric("Skills Match", f"{score_data.get('skills_match', 0):.1f}/40")
                st.metric("Experience", f"{score_data.get('experience_match', 0):.1f}/25")
                st.metric("Education", f"{score_data.get('education_match', 0):.1f}/15")
            
            with col2:
                # Radar chart
                breakdown = {
                    "Academic": score_data.get("academic_score", 0),
                    "Skills": score_data.get("skills_match", 0),
                    "Experience": score_data.get("experience_match", 0),
                    "Education": score_data.get("education_match", 0)
                }
                fig = create_readiness_radar(breakdown)
                st.plotly_chart(fig, use_container_width=True)
            
            # Gaps and Strengths
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ⚠️ Skill Gaps")
                gaps = score_data.get("gaps", [])
                if gaps:
                    for gap in gaps[:5]:
                        st.write(f"• {gap}")
                else:
                    st.info("No major gaps identified!")
            
            with col2:
                st.markdown("#### ✅ Strengths")
                strengths = score_data.get("strengths", [])
                if strengths:
                    for strength in strengths[:5]:
                        st.write(f"• {strength}")
                else:
                    st.info("Analyzing strengths...")
        
        st.divider()
        
        # Skill Gap Analysis
        if st.button("🔍 Analyze Skill Gaps", type="primary"):
            with st.spinner("Analyzing skill gaps..."):
                try:
                    resume_text = " ".join(st.session_state.pdf_texts.values()) if has_resume else ""
                    gap_data = analyze_skill_gaps(resume_text, target_role)
                    st.session_state.skill_gaps = gap_data
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if "skill_gaps" in st.session_state:
            gap_data = st.session_state.skill_gaps
            user_skills = gap_data.get("user_skills", [])
            required_skills = gap_data.get("required_skills", [])
            missing_skills = gap_data.get("missing_skills", [])
            
            if user_skills or required_skills:
                fig = create_skill_gap_chart(user_skills, required_skills)
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Skills", len(user_skills))
            with col2:
                st.metric("Required Skills", len(required_skills))
            with col3:
                st.metric("Missing Skills", len(missing_skills))
            
            if missing_skills:
                st.markdown("#### Missing Skills")
                for skill in missing_skills[:10]:
                    st.write(f"• {skill}")
        
        st.divider()
        
        # Performance Trends
        if has_marks:
            st.markdown("### 📈 Performance Trends")
            trends = analyze_performance_trends(combined_csv)
            
            if "error" not in trends:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Average", f"{trends.get('overall_average', 0):.1f}")
                with col2:
                    st.metric("Best Semester", f"Sem {trends.get('best_semester', 0)}")
                with col3:
                    st.metric("Trend", trends.get("trend_direction", "stable").title())
                with col4:
                    st.metric("Next Sem Prediction", f"{trends.get('next_semester_prediction', 0):.1f}")
                
                # Trend chart
                if "semester_averages" in trends:
                    import pandas as pd
                    trend_df = pd.DataFrame({
                        "Semester": range(1, len(trends["semester_averages"]) + 1),
                        "Average": trends["semester_averages"]
                    })
                    fig = create_interactive_chart(trend_df, "line", "Semester", "Average", 
                                                  "Performance Trend", "Semester", "Average Marks")
                    st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Recommendations
        if st.button("💡 Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                try:
                    resume_text = " ".join(st.session_state.pdf_texts.values()) if has_resume else ""
                    skill_gaps = st.session_state.get("skill_gaps", {}).get("skill_gaps", [])
                    recommendations = get_recommendations(resume_text, combined_csv, target_role, skill_gaps)
                    st.session_state.recommendations = recommendations
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if "recommendations" in st.session_state:
            st.markdown("### 💡 Personalized Recommendations")
            recs = st.session_state.recommendations
            for i, rec in enumerate(recs[:10], 1):
                with st.expander(f"{i}. {rec.get('title', 'Recommendation')} ({rec.get('priority', 'medium')} priority)"):
                    st.write(rec.get("description", ""))
                    if rec.get("resource"):
                        st.write(f"**Resource:** {rec['resource']}")
    
    except Exception as e:
        st.error(f"Career analysis error: {e}")


def main():
    # Initialize session state
    for key, default in [
        ("messages", []),
        ("processed_files", set()),
        ("pdf_texts", {}),
        ("csv_dataframes", {}),
        ("csv_bytes", {}),
        ("msg_counter", 0),
        ("last_audio_id", None),
        ("heard_text", ""),
        ("user_id", None),
        ("selected_language", "en"),
        ("conversation_history", []),
        ("bookmarks", []),
        ("current_profile", "default"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Restore CSVs from disk after page refresh
    import io, os, glob
    for fpath in glob.glob("chroma_db/csv_*.csv") + glob.glob("chroma_db/csv_*.CSV"):
        name = os.path.basename(fpath)[4:]  # strip "csv_" prefix
        if name not in st.session_state.csv_dataframes:
            try:
                raw = open(fpath, "rb").read()
                st.session_state.csv_bytes[name] = raw
                st.session_state.csv_dataframes[name] = load_csv(io.BytesIO(raw))
            except Exception:
                pass

    # Auto-load sample test files on first visit (so interviewers can test immediately)
    _preload_done_key = "_preload_done"
    if not st.session_state.get(_preload_done_key):
        st.session_state[_preload_done_key] = True
        sample_dir = os.path.join(os.path.dirname(__file__), "tests")
        preloaded = False
        for fname in ["arjun_sharma_resume.pdf", "arjun_sharma_career_goal.pdf", "btech_marks.csv"]:
            fpath = os.path.join(sample_dir, fname)
            if not os.path.exists(fpath) or fname in st.session_state.processed_files:
                continue
            try:
                if fname.endswith(".csv"):
                    raw = open(fpath, "rb").read()
                    st.session_state.csv_bytes[fname] = raw
                    st.session_state.csv_dataframes[fname] = load_csv(io.BytesIO(raw))
                    os.makedirs("chroma_db", exist_ok=True)
                    open(f"chroma_db/csv_{fname}", "wb").write(raw)
                else:
                    raw = open(fpath, "rb").read()
                    chunks = parse_pdf(io.BytesIO(raw), source_name=fname)
                    add_chunks(chunks)
                    full_text = " ".join(c["text"] for c in chunks)
                    st.session_state.pdf_texts[fname] = full_text
                st.session_state.processed_files.add(fname)
                preloaded = True
            except Exception:
                pass
        if preloaded:
            st.rerun()

    mode, audio = render_sidebar()
    has_docs = bool(get_stored_sources())

    # Merge all CSVs into one DataFrame if multiple uploaded
    csv_frames = list(st.session_state.csv_dataframes.values())
    if len(csv_frames) == 1:
        combined_csv = csv_frames[0]
    elif len(csv_frames) > 1:
        combined_csv = pd.concat(csv_frames, ignore_index=True)
    else:
        combined_csv = None

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 Chat", "🎯 Career Analysis", "⚙️ Settings"])

    with tab1:
        render_dashboard(combined_csv)

    with tab3:
        render_career_analysis(combined_csv)
    
    with tab4:
        st.markdown("### ⚙️ Settings & Tools")
        
        # Language Selection
        st.markdown("#### 🌐 Language Settings")
        selected_lang = st.selectbox("Response Language", 
                                    ["en", "es", "fr", "de", "hi", "zh", "ja"],
                                    index=0,
                                    format_func=lambda x: {"en": "English", "es": "Spanish", "fr": "French", 
                                                          "de": "German", "hi": "Hindi", "zh": "Chinese", "ja": "Japanese"}.get(x, x))
        st.session_state.selected_language = selected_lang
        
        st.divider()
        
        # Conversation History
        st.markdown("#### 💾 Conversation History")
        user_id = get_user_id()
        
        if st.button("💾 Save Current Conversation"):
            try:
                save_conversation(user_id, {
                    "title": f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "messages": st.session_state.messages,
                    "metadata": {"file_count": len(st.session_state.processed_files)}
                })
                st.success("✅ Conversation saved!")
            except Exception as e:
                st.error(f"Error saving: {e}")
        
        if st.button("📂 Load Conversations"):
            try:
                conversations = load_conversations(user_id)
                st.session_state.conversation_history = conversations
                st.success(f"✅ Loaded {len(conversations)} conversations")
            except Exception as e:
                st.error(f"Error loading: {e}")
        
        if st.session_state.get("conversation_history"):
            st.markdown("**Saved Conversations:**")
            for conv in st.session_state.conversation_history[:5]:
                if st.button(f"📄 {conv.get('title', 'Untitled')}", key=f"load_{conv.get('id')}"):
                    st.session_state.messages = conv.get("messages", [])
                    st.rerun()
        
        st.divider()
        
        # User Profiles
        st.markdown("#### 👤 User Profiles")
        profile_name = st.text_input("Profile Name", value="default")
        career_path = st.text_input("Career Path", placeholder="e.g., Data Science, Software Engineering")
        
        if st.button("💾 Save Profile"):
            try:
                save_user_profile(user_id, {
                    "name": profile_name,
                    "career_path": career_path,
                    "pdf_texts": st.session_state.pdf_texts,
                    "csv_dataframes": {k: v.to_dict() for k, v in st.session_state.csv_dataframes.items()}
                })
                st.success("✅ Profile saved!")
            except Exception as e:
                st.error(f"Error: {e}")
        
        if st.button("📂 Load Profiles"):
            try:
                profiles = load_user_profiles(user_id)
                st.session_state.user_profiles = profiles
                st.success(f"✅ Loaded {len(profiles)} profiles")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.divider()
        
        # Bookmarks
        st.markdown("#### 🔖 Bookmarks")
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if last_msg.get("role") == "assistant":
                if st.button("🔖 Bookmark Last Response"):
                    try:
                        save_bookmark(user_id, {
                            "title": f"Bookmark {datetime.now().strftime('%Y-%m-%d')}",
                            "content": last_msg.get("content", ""),
                            "source": last_msg.get("source", ""),
                            "metadata": {"message_id": last_msg.get("id")}
                        })
                        st.success("✅ Bookmarked!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        if st.button("📋 View Bookmarks"):
            try:
                bookmarks = load_bookmarks(user_id)
                st.session_state.bookmarks = bookmarks
            except Exception as e:
                st.error(f"Error: {e}")
        
        if st.session_state.get("bookmarks"):
            for bm in st.session_state.bookmarks[:5]:
                with st.expander(bm.get("title", "Bookmark")):
                    st.write(bm.get("content", ""))
        
        st.divider()
        
        # Export Options
        st.markdown("#### 📥 Export Chat")
        export_format = st.selectbox("Export Format", ["Markdown", "HTML", "PDF", "Word"])
        
        if st.button("⬇️ Export"):
            try:
                if export_format == "Markdown":
                    content = export_chat_markdown(st.session_state.messages)
                    st.download_button("Download", content, "chat.md", "text/markdown")
                elif export_format == "HTML":
                    content = export_chat_html(st.session_state.messages)
                    st.download_button("Download", content, "chat.html", "text/html")
                elif export_format == "PDF":
                    content = export_chat_pdf(st.session_state.messages)
                    st.download_button("Download", content, "chat.pdf", "application/pdf")
                elif export_format == "Word":
                    content = export_chat_word(st.session_state.messages)
                    st.download_button("Download", content, "chat.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Export error: {e}")
        
        st.divider()
        
        # Cache Management
        st.markdown("#### 🗄️ Cache Management")
        if st.button("🗑️ Clear Cache"):
            clear_cache()
            st.success("✅ Cache cleared!")
    
    with tab2:
        if not st.session_state.messages:
            st.markdown("""
<div class="intro-box">
  <h2 style="margin:0 0 8px 0; color:#e2e8f0; font-size:28px; font-weight:700;">🔬 Welcome to Analyser Bot</h2>
  <p style="margin:0 0 24px 0; color:#94a3b8; font-size:16px; line-height:1.6;">
    Your personal career intelligence assistant — upload your marks, resume, and goals,
    then ask anything. Get data-driven answers with charts, citations, and live job market insights.
  </p>

  <div class="feature-grid">
    <div class="feature-card">
      <h4>📊 Data Analysis</h4>
      <p>Upload semester mark CSVs. Ask for averages, rankings, trends — SQL runs automatically and charts are generated.</p>
    </div>
    <div class="feature-card">
      <h4>📄 Document Q&A</h4>
      <p>Upload your resume or goal statement as PDF. Ask what skills you have, what experience is listed, or what your goals say.</p>
    </div>
    <div class="feature-card">
      <h4>🌐 Live Web Search</h4>
      <p>Ask about job requirements, company expectations, or industry trends — Analyser Bot searches the web in real time.</p>
    </div>
    <div class="feature-card">
      <h4>🤖 Smart Routing</h4>
      <p>The AI decides which source to use — data, documents, web, or all three — based on your question. No manual switching.</p>
    </div>
    <div class="feature-card">
      <h4>🎙️ Voice I/O</h4>
      <p>Speak your question using the mic in the sidebar. Click 🔊 Listen on any response to hear it read aloud.</p>
    </div>
    <div class="feature-card">
      <h4>⚡ Concise / Detailed</h4>
      <p>Toggle between a quick 2-sentence answer or a full in-depth explanation — your choice, every time.</p>
    </div>
  </div>

  <div style="margin-top:28px; padding:20px; background:linear-gradient(135deg, #1a1f2e, #0f1419); border-radius:12px; border:1px solid #2a3a5a;">
    <p style="color:#60a5fa; font-size:13px; font-weight:600; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;">💡 Try Asking</p>
    <div style="display:flex; flex-wrap:wrap; gap:8px;">
      <span class="query-chip">What is my average CGPA across all semesters?</span>
      <span class="query-chip">Which semester did I score the highest?</span>
      <span class="query-chip">What skills are listed in my resume?</span>
      <span class="query-chip">What does Google require for a Data Engineer role?</span>
      <span class="query-chip">Am I ready for a Data Science internship?</span>
      <span class="query-chip">What should I improve to get into product companies?</span>
    </div>
  </div>

  <div style="margin-top:24px;">
    <p style="color:#60a5fa; font-size:13px; font-weight:600; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;">🚀 How to Get Started</p>
    <div class="step-row">
      <div class="step"><strong>Step 1 — Upload</strong>Add your semester marks CSV and/or resume PDF using the Dashboard tab.</div>
      <div class="step"><strong>Step 2 — Ask</strong>Type or speak your question in the chat box below. Be specific for best results.</div>
      <div class="step"><strong>Step 3 — Explore</strong>View charts, expand SQL queries, click sources, and listen to answers.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("### 💬 Chat")
        st.caption("Ask about your data, documents, or anything career-related. Use voice input from the sidebar for hands-free questions.")
        
        # Query Templates
        st.markdown("#### 📋 Quick Questions")
        template_cols = st.columns(4)
        templates = [
            "What is my average marks?",
            "What skills are in my resume?",
            "What does a Data Engineer need?",
            "Am I ready for this role?"
        ]
        for i, template in enumerate(templates):
            with template_cols[i]:
                if st.button(template, key=f"template_{i}", use_container_width=True):
                    st.session_state.template_query = template
                    st.rerun()
        
        # Search in chat history
        search_query = st.text_input("🔍 Search in chat history", key="chat_search")
        if search_query:
            matching = [msg for msg in st.session_state.messages if search_query.lower() in msg.get("content", "").lower()]
            if matching:
                st.info(f"Found {len(matching)} matching messages")
                for msg in matching[:3]:
                    with st.expander(f"{msg.get('role').capitalize()}: {msg.get('content', '')[:50]}..."):
                        st.write(msg.get("content", ""))
        
        # Handle template query
        if st.session_state.get("template_query"):
            query = st.session_state.template_query
            del st.session_state.template_query
            # Trigger the query
            if query:
                st.session_state.msg_counter += 1
                user_msg = {"role": "user", "content": query, "id": st.session_state.msg_counter}
                st.session_state.messages.append(user_msg)
                st.rerun()

        for msg in st.session_state.messages:
            render_message(msg)

        user_input = st.chat_input("Ask anything...")

        voice_query = ""
        if audio and audio.get("bytes") and audio.get("id") != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio.get("id")
            with st.spinner("Transcribing..."):
                try:
                    voice_query = speech_to_text(audio["bytes"])
                    if voice_query:
                        st.session_state.heard_text = voice_query
                except Exception as e:
                    st.error(f"Transcription error: {e}")

        if st.session_state.get("heard_text") and not voice_query:
            st.session_state.heard_text = ""

        if voice_query:
            st.info(f"🎙️ Heard: *{voice_query}*")

        user_input = user_input or voice_query

        if user_input:
            # Language translation
            translated_query, detected_lang = translate_query(user_input)
            
            # Check cache
            cached_response = get_cached_response(translated_query, {"has_docs": has_docs, "has_csv": combined_csv is not None})
            
            st.session_state.msg_counter += 1
            user_msg = {"role": "user", "content": user_input, "id": st.session_state.msg_counter}
            st.session_state.messages.append(user_msg)

            with st.chat_message("user"):
                st.markdown(user_input)
                if detected_lang != "en":
                    st.caption(f"🌐 Detected: {detected_lang}")

            with st.chat_message("assistant"):
                if cached_response:
                    st.info("💾 Using cached response")
                    result = cached_response
                else:
                    with st.spinner("Thinking..."):
                        result = ask(
                            query=translated_query,
                            history=st.session_state.messages[:-1],
                            mode=mode,
                            has_docs=has_docs,
                            csv_df=combined_csv,
                        )
                    # Cache the response
                    cache_response(translated_query, result, {"has_docs": has_docs, "has_csv": combined_csv is not None})
                
                source = result["source"]
                
                # Translate response if needed
                response_text = result["answer"]
                if st.session_state.selected_language != "en":
                    response_text = translate_response(response_text, st.session_state.selected_language)

                # Save message first so id is stable across reruns
                st.session_state.msg_counter += 1
                ai_msg = {
                    "role": "assistant",
                    "content": response_text,
                    "source": source,
                    "id": st.session_state.msg_counter,
                    "sql_result": result.get("sql_result"),
                }
                st.session_state.messages.append(ai_msg)

                st.markdown(BADGE.get(source, BADGE["llm"]), unsafe_allow_html=True)
                st.markdown(response_text)
                if result.get("sql_result"):
                    render_sql_result(result["sql_result"], ai_msg["id"])
                
                # Follow-up suggestions
                if st.session_state.get("enable_followups", True):
                    st.markdown("**💡 Suggested Follow-ups:**")
                    suggestions = [
                        "Tell me more about this",
                        "What are the next steps?",
                        "How can I improve this?",
                        "Show me examples"
                    ]
                    cols = st.columns(len(suggestions))
                    for i, sug in enumerate(suggestions):
                        with cols[i]:
                            if st.button(sug, key=f"followup_{i}"):
                                st.session_state.followup_query = sug
                                st.rerun()
                
                st.rerun()
        
        # Handle follow-up query
        if st.session_state.get("followup_query"):
            query = st.session_state.followup_query
            del st.session_state.followup_query
            # Trigger the query by setting it as user input
            if query:
                st.session_state.msg_counter += 1
                user_msg = {"role": "user", "content": query, "id": st.session_state.msg_counter}
                st.session_state.messages.append(user_msg)
                st.rerun()


if __name__ == "__main__":
    main()
