"""
LayoutLens — Structure-Aware Document OCR & Table Extractor

Upload a document image, run word-level OCR (PaddleOCR locally or Azure AI Vision),
and view the output side-by-side as a layout-preserved HTML structure or raw OCR text.
"""
import os
import tempfile
import time
import traceback

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from html_table_builder import build_html_table, group_words_into_rows
from recognizers.azure_recognizer import AzureCredentialsError, AzureTextRecognizer
from recognizers.paddle_recognizer import PaddleTextRecognizer

load_dotenv()

# Free-tier (F0) guardrail for Azure tab: Azure allows 20 transactions/minute; self-limit to 15
AZURE_MAX_CALLS_PER_MINUTE = 15


def check_azure_rate_limit() -> bool:
    """Return True if another Azure call is allowed right now."""
    now = time.time()
    history = [t for t in st.session_state.get('azure_call_times', []) if now - t < 60]
    st.session_state.azure_call_times = history
    return len(history) < AZURE_MAX_CALLS_PER_MINUTE


def record_azure_call() -> None:
    st.session_state.setdefault('azure_call_times', []).append(time.time())


st.set_page_config(
    page_title="LayoutLens | Spatial Document OCR",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .main { padding: 1.5rem; }
        .stApp { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); }

        .main-header {
            background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
            color: white;
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .main-header h1 { font-size: 2.2rem; margin-bottom: 0.4rem; font-weight: 700; color: #ffffff; }
        .main-header p { font-size: 1.05rem; opacity: 0.92; margin: 0; }

        .section-header {
            font-size: 1.15rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 0.75rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid #e2e8f0;
        }

        .feature-card {
            background-color: #ffffff;
            padding: 0.9rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #2563eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            font-size: 0.92rem;
            color: #334155;
            line-height: 1.5;
        }

        .azure-stats {
            background-color: #eff6ff;
            color: #1e40af;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid #bfdbfe;
            font-size: 0.9rem;
        }

        .guide-text { color: #475569; font-size: 0.9rem; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧩 LayoutLens Guide")
    st.markdown("""
        <div class="guide-text">
            <strong>LayoutLens</strong> reconstructs document geometry directly from raw OCR bounding boxes:
            <ol style="padding-left: 1.2rem; margin-top: 0.5rem;">
                <li>Select an OCR engine (<strong>PaddleOCR</strong> for offline speed, or <strong>Azure AI Vision</strong>).</li>
                <li>Upload any document image (invoice, form, table, or receipt).</li>
                <li>Inspect the side-by-side comparison: original source on the left, spatial HTML grid on the right.</li>
                <li>Export the output as an HTML layout or raw word-level text.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("☁️ Azure AI Vision Setup", expanded=False):
        st.markdown("""
            <div class="guide-text">
                The Azure tab runs the <strong>Computer Vision Read</strong> API.
            </div>
            <div class="feature-card">
                <strong>Quick Setup:</strong>
                <ol style="padding-left: 1.2rem; margin: 0.5rem 0 0 0;">
                    <li>Sign into <a href="https://portal.azure.com" target="_blank">Azure Portal</a>.</li>
                    <li>Create a <strong>Computer Vision</strong> resource on tier <strong>Free F0</strong>.</li>
                    <li>Paste your Key and Endpoint in the Azure tab or set them in <code>.env</code>.</li>
                </ol>
            </div>
            <div class="guide-text">
                <strong>Free-tier Guardrail:</strong> Self-limited to {max_calls} calls/minute (Azure limit: 20/min).
            </div>
        """.format(max_calls=AZURE_MAX_CALLS_PER_MINUTE), unsafe_allow_html=True)

if 'is_expanded' not in st.session_state:
    st.session_state.is_expanded = False

if 'paddle_recognizer' not in st.session_state:
    st.session_state.paddle_recognizer = PaddleTextRecognizer()

# Top Hero Banner
st.markdown("""
    <div class="main-header">
        <h1>🧩 LayoutLens</h1>
        <p>Structure-Aware Document OCR & Spatial Table Extractor — Convert document images into layout-preserved HTML</p>
    </div>
""", unsafe_allow_html=True)


def render_results(prefix: str, uploaded_file, raw_df, html_table):
    """
    Renders a side-by-side view:
    - Left column: The uploaded document image.
    - Right column: Tabbed/toggle outputs (Rendered HTML, Raw OCR, Code).
    """
    raw_text = "\n".join(
        " ".join(word for word, _bbox in row)
        for row in group_words_into_rows(raw_df)
    )
    st.session_state[f'raw_data_{prefix}'] = raw_df
    st.session_state[f'raw_text_{prefix}'] = raw_text
    st.session_state[f'html_table_{prefix}'] = html_table

    st.markdown("---")
    col_left, col_right = st.columns([1, 1], gap="large")

    # Left Column: Uploaded Document Visual
    with col_left:
        st.markdown('<div class="section-header">📄 Original Document</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True, caption=f"Uploaded: {uploaded_file.name}")

    # Right Column: Spatial Output Views
    with col_right:
        st.markdown('<div class="section-header">🎯 LayoutLens Output</div>', unsafe_allow_html=True)

        output_tab1, output_tab2, output_tab3 = st.tabs([
            "🌐 Rendered Layout",
            "📝 Raw OCR Text",
            "📋 HTML Code"
        ])

        with output_tab1:
            st.markdown("""
                <div class="feature-card">
                    Bounding boxes are mapped onto a uniform geometric grid. Cells spanned by multi-word headers
                    or fields are merged dynamically (<code>colspan</code>/<code>rowspan</code>) to mirror the original structure.
                </div>
            """, unsafe_allow_html=True)

            components.html(
                f"<div style='overflow:auto; font-family: sans-serif; background:#fff; padding:10px; border-radius:6px; border: 1px solid #e2e8f0;'>{html_table}</div>",
                height=550 if not st.session_state.is_expanded else 900,
                scrolling=True,
            )

            st.download_button(
                label="📥 Download Preserved HTML",
                data=html_table,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_layout.html",
                mime="text/html",
                key=f"html_download_{prefix}",
                use_container_width=True,
            )

        with output_tab2:
            st.text_area(
                "Extracted Text Stream",
                value=raw_text,
                height=480 if not st.session_state.is_expanded else 750,
                disabled=True,
                key=f"raw_text_area_{prefix}",
            )
            st.download_button(
                label="📥 Download Raw Text (.txt)",
                data=raw_text,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_raw.txt",
                mime="text/plain",
                key=f"raw_download_{prefix}",
                use_container_width=True,
            )

        with output_tab3:
            st.markdown("##### Underlying HTML Structure")
            st.code(html_table, language="html")


# Engine Tabs
engine_tabs = st.tabs(["⚡ PaddleOCR (Local Engine)", "☁️ Azure AI Vision (Cloud Engine)"])

# ---------------------------------------------------------------------------
# PaddleOCR tab — runs locally
# ---------------------------------------------------------------------------
with engine_tabs[0]:
    uploaded_file_paddle = st.file_uploader(
        "Upload a document image for local layout parsing",
        type=['png', 'jpg', 'jpeg'],
        key="uploader_paddle",
        help="Runs fully on your machine using PaddleOCR models.",
    )

    if uploaded_file_paddle is not None:
        with st.spinner('🔄 Analyzing document geometry and running PaddleOCR...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file_paddle.getvalue())
                temp_path = tmp_file.name

            try:
                raw_df = st.session_state.paddle_recognizer.recognize(temp_path)[0]
                html_table = build_html_table(raw_df)
                render_results("paddle", uploaded_file_paddle, raw_df, html_table)

            except Exception:
                st.error(f"❌ Error processing image: {traceback.format_exc()}")

            finally:
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    st.warning(f"⚠️ Error removing temporary file: {str(e)}")

# ---------------------------------------------------------------------------
# Azure AI Vision tab — cloud OCR
# ---------------------------------------------------------------------------
with engine_tabs[1]:
    env_endpoint = os.environ.get('AZURE_CV_ENDPOINT', '')
    env_key = os.environ.get('AZURE_CV_KEY', '')

    with st.expander("🔑 Azure Credentials & Settings", expanded=not (env_endpoint and env_key)):
        st.markdown(
            "Enter credentials below or set `AZURE_CV_ENDPOINT` and `AZURE_CV_KEY` in your `.env` file."
        )
        manual_endpoint = st.text_input(
            "Endpoint URL",
            key="manual_azure_endpoint",
            placeholder=env_endpoint or "https://<resource>.cognitiveservices.azure.com/",
        )
        manual_key = st.text_input(
            "Subscription Key",
            key="manual_azure_key",
            type="password",
            placeholder="Using .env value" if env_key else "32-character key",
        )

    azure_endpoint = manual_endpoint.strip() or env_endpoint
    azure_key = manual_key.strip() or env_key

    if not (azure_endpoint and azure_key):
        st.info(
            "☁️ Azure AI Vision Read API requires an Azure subscription key and endpoint. "
            "Configure them above or switch to the **PaddleOCR** tab for local execution."
        )
    else:
        creds = (azure_endpoint, azure_key)
        if st.session_state.get('azure_recognizer_creds') != creds:
            try:
                st.session_state.azure_recognizer = AzureTextRecognizer(
                    endpoint=azure_endpoint, subscription_key=azure_key
                )
                st.session_state.azure_recognizer_error = None
            except AzureCredentialsError as e:
                st.session_state.azure_recognizer = None
                st.session_state.azure_recognizer_error = str(e)
            st.session_state.azure_recognizer_creds = creds

        if st.session_state.get('azure_recognizer') is None:
            st.error(f"❌ {st.session_state.get('azure_recognizer_error', 'Azure client could not be initialized.')}")
        else:
            uploaded_file_azure = st.file_uploader(
                "Upload a document image for Azure cloud parsing",
                type=['png', 'jpg', 'jpeg'],
                key="uploader_azure",
                help=f"Azure's free tier allows {AZURE_MAX_CALLS_PER_MINUTE} calls/minute in this app.",
            )

            if uploaded_file_azure is not None:
                if not check_azure_rate_limit():
                    st.warning(
                        f"⏳ Rate limit reached ({AZURE_MAX_CALLS_PER_MINUTE} calls/minute). "
                        "Please wait a few moments before submitting again."
                    )
                else:
                    with st.spinner('🔄 Calling Azure AI Vision Read & reconstructing layout...'):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_file_azure.getvalue())
                            temp_path = tmp_file.name

                        try:
                            record_azure_call()
                            raw_df = st.session_state.azure_recognizer.recognize(temp_path)[0]
                            html_table = build_html_table(raw_df)

                            stats = st.session_state.azure_recognizer.last_call_stats
                            if stats:
                                st.markdown(f"""
                                    <div class="azure-stats">
                                        ⚡ Azure Read: <strong>{stats.get('elapsed_seconds', '?')}s</strong> ·
                                        Status: <strong>{stats.get('status', '?')}</strong> ·
                                        Rotation detected: <strong>{stats.get('rotation_angle', 0):.2f}°</strong>
                                    </div>
                                """, unsafe_allow_html=True)

                            render_results("azure", uploaded_file_azure, raw_df, html_table)

                        except Exception:
                            st.error(f"❌ Error processing image: {traceback.format_exc()}")

                        finally:
                            try:
                                os.unlink(temp_path)
                            except Exception as e:
                                st.warning(f"⚠️ Error removing temporary file: {str(e)}")