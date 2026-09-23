import base64
import os
import tempfile

import numpy as np
import streamlit as st

from openai import OpenAI
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InteriorIQ AI Studio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SHOWROOM CSS — black chassis, amber spotlight glow
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --void: #060607;
        --panel: #111114;
        --panel-2: #17171b;
        --line: rgba(255,255,255,0.07);
        --line-strong: rgba(255,255,255,0.14);
        --amber: #ffb400;
        --amber-soft: #ffcf6b;
        --amber-glow: rgba(255,180,0,0.35);
        --chrome: #f3f3f1;
        --grey: #93949c;
        --grey-dim: #5c5d64;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }


    /* ---------- GLOBAL / SHOWROOM FLOOR ---------- */

    .stApp {
        background:
            radial-gradient(ellipse 900px 500px at 20% -5%, rgba(255,180,0,0.10), transparent 60%),
            radial-gradient(ellipse 700px 450px at 85% 0%, rgba(255,180,0,0.06), transparent 55%),
            linear-gradient(180deg, #060607 0%, #08080a 55%, #0a0a0c 100%);
        color: var(--chrome);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #08080a;
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--line);
    }


    /* ---------- HERO — showroom spotlight ---------- */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 3rem 2.6rem 2.6rem;
        border-radius: 4px;
        background: linear-gradient(180deg, #131316 0%, #0c0c0e 100%);
        border: 1px solid var(--line);
        border-top: 1px solid var(--line-strong);
        margin-bottom: 1.8rem;
        box-shadow:
            0 40px 90px rgba(0,0,0,0.55),
            inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .hero::before {
        /* the overhead spotlight beam */
        content: "";
        position: absolute;
        top: -260px;
        left: 8%;
        width: 480px;
        height: 480px;
        background: conic-gradient(
            from 200deg at 50% 0%,
            transparent 0deg,
            rgba(255,180,0,0.16) 18deg,
            rgba(255,180,0,0.05) 34deg,
            transparent 46deg
        );
        filter: blur(2px);
        pointer-events: none;
    }

    .hero::after {
        /* amber ground glow under the beam */
        content: "";
        position: absolute;
        bottom: -140px;
        left: -60px;
        width: 420px;
        height: 260px;
        background: radial-gradient(ellipse, rgba(255,180,0,0.14), transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        position: relative;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
        background: linear-gradient(100deg, #ffffff 30%, var(--amber-soft) 70%, var(--amber) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-subtitle {
        position: relative;
        color: var(--grey);
        font-size: 1.05rem;
        max-width: 800px;
        line-height: 1.7;
    }

    .badge {
        position: relative;
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 3px;
        background: rgba(255,180,0,0.08);
        border: 1px solid rgba(255,180,0,0.35);
        color: var(--amber-soft);
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 1.1rem;
    }


    /* ---------- CARDS — brushed chrome panels ---------- */

    .card {
        position: relative;
        background: linear-gradient(165deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow:
            0 18px 45px rgba(0,0,0,0.35),
            inset 0 1px 0 rgba(255,255,255,0.05);
        transition: box-shadow 0.25s ease, border-color 0.25s ease;
    }

    .card:hover {
        border-color: rgba(255,180,0,0.25);
        box-shadow:
            0 18px 45px rgba(0,0,0,0.4),
            0 0 0 1px rgba(255,180,0,0.08),
            inset 0 1px 0 rgba(255,255,255,0.06);
    }

    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        color: var(--chrome);
    }

    .card-subtitle {
        color: var(--grey);
        font-size: 0.88rem;
    }


    /* ---------- METRICS — dashboard gauges ---------- */

    .metric-card {
        position: relative;
        background: radial-gradient(ellipse at 50% -20%, rgba(255,180,0,0.07), transparent 60%), #0b0b0d;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 1.1rem 1rem;
        text-align: center;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02), 0 10px 25px rgba(0,0,0,0.3);
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--amber-soft);
        text-shadow: 0 0 18px rgba(255,180,0,0.35);
    }

    .metric-label {
        color: var(--grey-dim);
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        margin-top: 0.3rem;
    }


    /* ---------- SECTION TITLES — showroom placard ---------- */

    .section-title {
        position: relative;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.6rem;
        margin-bottom: 0.9rem;
        padding-left: 0.9rem;
        border-left: 3px solid var(--amber);
        color: var(--chrome);
    }


    /* ---------- BUTTON — matte black, amber underglow ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 4px;
        border: 1px solid rgba(255,180,0,0.4);
        background: linear-gradient(180deg, #1a1a1d 0%, #0c0c0e 100%);
        color: var(--amber-soft);
        font-weight: 600;
        letter-spacing: 0.01em;
        padding: 0.8rem 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        color: #ffffff;
        border-color: var(--amber);
        transform: translateY(-2px);
        box-shadow:
            0 16px 40px rgba(0,0,0,0.5),
            0 0 24px rgba(255,180,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.08);
    }


    /* ---------- UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px dashed var(--line-strong);
        border-radius: 4px;
    }


    /* ---------- INFO ---------- */

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 4px;
        background: linear-gradient(180deg, rgba(255,180,0,0.06), rgba(255,180,0,0.02));
        border: 1px solid rgba(255,180,0,0.18);
        border-left: 3px solid var(--amber);
        color: #cfd0d6;
        line-height: 1.6;
    }


    /* ---------- CODE / RGB SWATCH ---------- */

    .stCodeBlock, code {
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: var(--grey-dim);
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--line);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API / MODEL SETUP
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Please set your API key as an environment variable."
    )
    st.stop()

client = OpenAI(api_key=api_key)


@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")


model = load_yolo_model()


# ============================================================
# HERO SECTION
# ============================================================

st.title("🏠 AI Interior Design Studio")
st.caption(
    "AI-powered room analysis, design recommendations, and visual redesign"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Design Studio")

    st.markdown(
        """
        <div class="card-subtitle">
        Configure your redesign preferences before generating your concept.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    budget = st.number_input(
        "💰 Budget",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100
    )

    st.caption("Approximate redesign budget in USD")

    style = st.selectbox(
        "🎨 Interior Style",
        [
            "Modern",
            "Minimalist",
            "Boho",
            "Industrial",
            "Traditional"
        ]
    )

    color_pref = st.text_input(
        "🌈 Preferred Color",
        placeholder="e.g. Beige, navy blue..."
    )

    st.markdown("---")

    st.markdown("### 🤖 AI Pipeline")

    st.markdown(
        """
        **01** Image Analysis  
        **02** Object Detection  
        **03** Color Extraction  
        **04** AI Design Plan  
        **05** Visual Redesign
        """
    )

    st.markdown("---")

    st.caption("Powered by OpenAI + YOLOv8 + Streamlit")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📸 Upload Your Room</div>',
    unsafe_allow_html=True
)

img_file = st.file_uploader(
    "Upload a room photo",
    type=["jpg", "jpeg", "png"],
    help="For best results, upload a clear room photo."
)


if not img_file:

    st.markdown(
        """
        <div class="info-box">
        👆 Upload a room photo to begin your AI-powered interior
        redesign journey.
        <br><br>
        <b>Recommended:</b> Clear JPG/PNG image with good lighting
        and a visible room layout.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# IMAGE PREVIEW
# ============================================================

img = Image.open(img_file).convert("RGB")

preview_col1, preview_col2 = st.columns([1.5, 1])

with preview_col1:

    st.image(
        img,
        caption="Uploaded Room",
        width="stretch"
    )

with preview_col2:

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        ✨ Ready to Transform
        </div>

        <div class="card-subtitle">
        Your room image is ready for AI analysis.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">${budget:,}</div>
            <div class="metric-label">DESIGN BUDGET</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-value">{style}</div>
            <div class="metric-label">SELECTED STYLE</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN BUTTON
# ============================================================

st.markdown("")

generate = st.button(
    "✨ Analyze & Redesign My Room",
    type="primary"
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if generate:

    # --------------------------------------------------------
    # STEP 1: OBJECT DETECTION
    # --------------------------------------------------------

    with st.status(
        "🔍 Analyzing your room...",
        expanded=True
    ) as status:

        st.write("Detecting furniture and objects...")

        results = model(np.array(img))

        labels = list(
            set(
                model.names[int(c)]
                for c in results[0].boxes.cls
            )
        )

        st.write("✓ Object detection completed")

        # ----------------------------------------------------
        # STEP 2: COLOR EXTRACTION
        # ----------------------------------------------------

        st.write("Extracting dominant colors...")

        small = img.resize((100, 100))

        pixels = np.array(small).reshape(-1, 3)

        kmeans = KMeans(
            n_clusters=4,
            n_init=10,
            random_state=42
        )

        kmeans.fit(pixels)

        colors = [
            tuple(map(int, c))
            for c in kmeans.cluster_centers_
        ]

        st.write("✓ Color analysis completed")

        # ----------------------------------------------------
        # STEP 3: GPT DESIGN PLAN
        # ----------------------------------------------------

        st.write("Creating personalized design plan...")

        prompt = f"""
You are an expert interior designer.

Analyze the following room information.

Detected objects:
{labels}

Dominant RGB colors:
{colors}

Budget:
${budget}

Preferred style:
{style}

Preferred color:
{color_pref or "No specific preference"}

Create a concise but useful interior redesign plan.

Include:

1. Overall design concept
2. What to keep
3. What to remove or replace
4. Recommended color palette
5. 4-5 furniture/decor recommendations
6. Approximate price range for each item
7. Budget summary
8. Practical styling tips

Keep the recommendations realistic and suitable for the detected room.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        suggestion_text = (
            response.choices[0]
            .message.content
        )

        st.write("✓ AI design plan generated")

        status.update(
            label="✅ Room analysis completed",
            state="complete"
        )


    # ========================================================
    # ANALYSIS DASHBOARD
    # ========================================================

    st.markdown(
        '<div class="section-title">🔎 Room Analysis</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {len(labels)}
                </div>
                <div class="metric-label">
                    OBJECTS DETECTED
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    4
                </div>
                <div class="metric-label">
                    DOMINANT COLORS
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    ${budget:,}
                </div>
                <div class="metric-label">
                    DESIGN BUDGET
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # OBJECT DETECTION + COLORS
    # ========================================================

    left, right = st.columns([1.5, 1])

    with left:

        st.markdown(
            """
            <div class="section-title">
            🪑 Detected Objects
            </div>
            """,
            unsafe_allow_html=True
        )

        annotated = results[0].plot()

        st.image(
            annotated[:, :, ::-1],
            caption="YOLOv8 Object Detection",
            width="stretch"
        )

        if labels:
            st.write(
                "Detected:",
                ", ".join(labels)
            )
        else:
            st.write("No supported objects detected.")


    with right:

        st.markdown(
            """
            <div class="section-title">
            🎨 Color Palette
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("#### RGB Values")
        for color in colors:
            st.code(
                    f"RGB{color}",
                    language=None
                    )




    # ========================================================
    # AI DESIGN RECOMMENDATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 AI Design Recommendations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">
        {suggestion_text}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # BEFORE / AFTER
    # ========================================================

    st.markdown(
        '<div class="section-title">🖼️ AI Room Transformation</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The AI visualization attempts to preserve the original "
        "room layout, architecture, windows, and camera perspective."
    )

    before_col, after_col = st.columns(2)

    with before_col:

        st.markdown("### Original")

        st.image(
            img,
            width="stretch"
        )


    with after_col:

        st.markdown("### AI Redesign")

        with st.spinner(
            "🎨 Creating your redesigned room..."
        ):

            edit_prompt = f"""
Redesign this exact room in {style.lower()} interior design style.

IMPORTANT:
- Preserve the original room architecture.
- Preserve the walls.
- Preserve windows and doors.
- Preserve the camera angle.
- Preserve the overall room layout.
- Make the result realistic and photorealistic.
- Update furniture and decor.
- Improve lighting and visual harmony.
- Use a {color_pref or "complementary neutral"} color palette.
- Make the room feel premium and professionally designed.
- The design should feel consistent with an approximate
  budget of ${budget}.
- Do not change the room into a different type of room.
"""

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".png"
                ) as temp_file:

                    img.save(
                        temp_file.name,
                        format="PNG"
                    )

                    temp_path = temp_file.name


                with open(
                    temp_path,
                    "rb"
                ) as image_file:

                    result = client.images.edit(
                        model="gpt-image-1-mini",
                        image=image_file,
                        prompt=edit_prompt,
                        size="1024x1024"
                    )


                after_bytes = base64.b64decode(
                    result.data[0].b64_json
                )

                st.image(
                    after_bytes,
                    width="stretch"
                )

                st.success(
                    "✨ Your AI room concept is ready!"
                )


            except Exception as e:

                st.error(
                    f"Image generation failed: {e}"
                )

            finally:

                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)


    # ========================================================
    # PROJECT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Design Summary</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {style}
                </div>
                <div class="metric-label">
                    DESIGN STYLE
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    ${budget:,}
                </div>
                <div class="metric-label">
                    TARGET BUDGET
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {color_pref or "AI Selected"}
                </div>
                <div class="metric-label">
                    COLOR DIRECTION
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🏠 RoomAI Studio &nbsp;•&nbsp;
        Powered by OpenAI + YOLOv8 + Streamlit
        <br>
        AI-generated redesigns are conceptual visualizations
        and may not represent exact real-world results.
    </div>
    """,
    unsafe_allow_html=True
)