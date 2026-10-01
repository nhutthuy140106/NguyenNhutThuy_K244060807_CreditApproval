from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Credit Lab | Phân tích dữ liệu",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at 85% 0%, #10294a 0, #081321 34%, #081321 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { border-right: 1px solid #20324c; }
    .block-container { max-width: 1320px; padding-top: 1.2rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: -.025em; }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: .48rem .62rem; border-radius: 10px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #1a304e;
    }
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #263b58;
        border-radius: 18px;
    }
    [data-testid="stForm"] { background: #101f34; padding: 1.1rem 1.3rem; }
    [data-testid="stPlotlyChart"] {
        background: #101f34; border: 1px solid #263b58;
        border-radius: 18px; padding: .55rem;
    }
    [data-testid="stDataFrame"] { border: 1px solid #263b58; border-radius: 14px; }
    .topline {
        display: flex; align-items: center; justify-content: space-between;
        gap: 1rem; padding: .2rem 0 1.1rem; margin-bottom: 1.45rem;
        border-bottom: 1px solid #20324c;
    }
    .brand { display: flex; align-items: center; gap: .8rem; font-weight: 750; font-size: 1.08rem; }
    .brand-icon {
        display: inline-grid; place-items: center; width: 37px; height: 37px;
        color: #061627; background: linear-gradient(135deg, #64e1f3, #5b85ff);
        border-radius: 11px; font-size: 1.35rem;
    }
    .status {
        color: #a8c4db; font-size: .8rem; border: 1px solid #29415d;
        background: #13263e; border-radius: 999px; padding: .44rem .75rem;
        white-space: nowrap;
    }
    .status-dot { color: #5ee0b7; margin-right: .28rem; }
    .eyebrow {
        color: #6cddf7; font-size: .75rem; font-weight: 800;
        letter-spacing: .16em; text-transform: uppercase;
    }
    .hero {
        display: grid; grid-template-columns: minmax(0, 1fr) 240px;
        gap: 1.5rem; align-items: center; overflow: hidden;
        padding: 2.5rem 3rem; margin-bottom: 1.8rem;
        border: 1px solid #28446c; border-radius: 24px;
        background: linear-gradient(115deg, #132848, #12233d 58%, #18385a);
        box-shadow: 0 24px 60px rgba(0, 0, 0, .17);
    }
    .hero h1 {
        color: #f3f8ff; font-size: clamp(2.6rem, 5vw, 4.5rem);
        line-height: 1.05; margin: .85rem 0 1rem;
    }
    .hero p { color: #adc3dc; max-width: 620px; font-size: 1rem; line-height: 1.65; margin: 0; }
    .hero-art {
        height: 178px; border-radius: 20px; display: grid; place-items: center;
        background: radial-gradient(circle at 30% 20%, #3c7ba3, #193654 55%, #142b48);
        border: 1px solid #366080; position: relative;
    }
    .hero-art span { color: #8beafa; font-size: 5rem; font-weight: 800; line-height: 1; }
    .hero-art small {
        position: absolute; bottom: 18px; font-size: .68rem; letter-spacing: .15em;
        color: #b5cce0; font-weight: 800;
    }
    .section-head { margin: 2.2rem 0 1rem; }
    .section-head h2 { color: #f1f6ff; margin: .35rem 0 .2rem; font-size: 1.65rem; }
    .section-head p { color: #9fb3ca; margin: 0; font-size: .92rem; }
    .stat-card, .model-card, .step-card, .callout {
        border: 1px solid #263b58; border-radius: 18px;
        background: #101f34; box-shadow: 0 12px 32px rgba(0, 0, 0, .12);
    }
    .stat-card { padding: 1.25rem 1.3rem; min-height: 152px; }
    .stat-card .label { color: #aac1d8; font-size: .81rem; }
    .stat-card .value { color: #f4f8ff; font-size: clamp(1.8rem, 2.4vw, 2.65rem); font-weight: 760; margin: .4rem 0 .12rem; }
    .stat-card .note { color: #75d6e9; font-size: .75rem; }
    .model-card { padding: 1.35rem 1.5rem; }
    .model-card.best { border-color: #347d90; background: linear-gradient(145deg, #112d40, #101f34 70%); }
    .model-top { display: flex; align-items: start; justify-content: space-between; gap: 1rem; }
    .model-name { color: #f4f8ff; font-size: 1.25rem; font-weight: 750; }
    .chip { color: #85e4e7; background: #163e4a; border: 1px solid #2c6872; border-radius: 999px; font-size: .72rem; padding: .3rem .65rem; white-space: nowrap; }
    .model-accuracy { color: #f5f9ff; font-size: 2.25rem; font-weight: 780; margin: .95rem 0 .1rem; }
    .model-sub { color: #95abc3; font-size: .76rem; }
    .mini-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .65rem; margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid #29405b; }
    .mini-grid span { color: #9fb5cb; font-size: .7rem; display: block; }
    .mini-grid strong { color: #dfeeff; font-size: .98rem; }
    .step-card { padding: 1.1rem 1.25rem; min-height: 112px; }
    .step-num { color: #6cddf7; font-size: .76rem; font-weight: 800; }
    .step-card strong { display: block; margin: .3rem 0; color: #ecf4ff; }
    .step-card p { color: #91a9c2; font-size: .8rem; margin: 0; line-height: 1.45; }
    .callout { padding: 1.25rem 1.45rem; margin: 1rem 0; }
    .callout p { color: #a9bed5; margin: .25rem 0 0; font-size: .86rem; }
    .result-good { border-color: #2c746b; background: linear-gradient(135deg, #123a3c, #102337); }
    .result-bad { border-color: #765a68; background: linear-gradient(135deg, #3c2a3c, #102337); }
    .result-label { font-size: 1.45rem; font-weight: 760; color: #f3f8ff; }
    .small-note { color: #91a9c2; font-size: .79rem; line-height: 1.55; }
    .footer { border-top: 1px solid #20324c; color: #829bb3; font-size: .74rem; margin-top: 2.8rem; padding-top: 1.2rem; }
    @media (max-width: 800px) {
        .hero { grid-template-columns: 1fr; padding: 1.8rem; }
        .hero h1 { font-size: 2.4rem; }
        .hero-art { display: none; }
        .topline { align-items: start; }
        .status { display: none; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


BASE = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE / "credit_model.joblib")


@st.cache_data
def load_tables():
    return (
        pd.read_csv(BASE / "credit_data.csv"),
        pd.read_csv(BASE / "model_comparison.csv"),
    )


try:
    artifacts = load_model()
    df, comparison = load_tables()
except Exception as exc:
    st.error("Không tải được model hoặc dữ liệu. Kiểm tra ba file .joblib và .csv trong repo.")
    with st.expander("Chi tiết lỗi"):
        st.code(str(exc))
    st.stop()

model = artifacts["model"]
features = artifacts["feature_columns"]
numeric = artifacts["numeric_columns"]
categorical = artifacts["categorical_columns"]

encoder = (
    model.named_steps["preprocessor"]
    .named_transformers_["categorical"]
    .named_steps["encoder"]
)
categories = {col: list(values) for col, values in zip(categorical, encoder.categories_)}
positive_index = list(model.classes_).index("+")


def predict_records(raw):
    """Dùng chính pipeline đã huấn luyện; không fit lại trên dữ liệu nhập."""
    raw = raw.copy()
    raw.columns = raw.columns.astype(str).str.strip()
    if raw.columns.duplicated().any():
        raise ValueError("CSV có tên cột bị trùng.")
    missing = [col for col in features if col not in raw.columns]
    if missing:
        raise ValueError("Thiếu các cột: " + ", ".join(missing))
    if len(raw) == 0:
        raise ValueError("File chưa có dòng dữ liệu.")
    if len(raw) > 5000:
        raise ValueError("Vui lòng dùng tối đa 5.000 dòng mỗi lần.")

    data = raw[features].copy()
    data = data.replace(r"^\s*$", np.nan, regex=True).replace("?", np.nan)
    for col in numeric:
        converted = pd.to_numeric(data[col], errors="coerce")
        invalid = data[col].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Cột {col} có giá trị không phải số.")
        if np.isinf(converted.to_numpy(dtype=float)).any():
            raise ValueError(f"Cột {col} có giá trị vô hạn.")
        data[col] = converted
    for col in categorical:
        data[col] = data[col].map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
        invalid = data[col].notna() & ~data[col].isin(categories[col])
        if invalid.any():
            raise ValueError(f"Cột {col} có mã chưa được hỗ trợ. Mã hợp lệ: {', '.join(map(str, categories[col]))}.")

    result = data.copy()
    result["Du_doan"] = model.predict(data)
    result["Xac_suat_lop_duoc_duyet"] = model.predict_proba(data)[:, positive_index]
    return result


def section(kicker, title, description=""):
    st.markdown(
        f'<div class="section-head"><div class="eyebrow">{escape(kicker)}</div>'
        f'<h2>{escape(title)}</h2><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def stat_card(label, value, note):
    st.markdown(
        f'<div class="stat-card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(str(value))}</div>'
        f'<div class="note">{escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def model_card(row, selected):
    name = str(row["Mô hình"])
    tag = "MÔ HÌNH SỬ DỤNG" if selected else "ĐỐI CHIẾU"
    st.markdown(
        f'<div class="model-card {"best" if selected else ""}">'
        f'<div class="model-top"><div class="model-name">{escape(name)}</div>'
        f'<div class="chip">{tag}</div></div>'
        f'<div class="model-accuracy">{float(row["Accuracy"]):.1%}</div>'
        '<div class="model-sub">Accuracy trên 138 hồ sơ kiểm tra</div>'
        '<div class="mini-grid">'
        f'<div><span>PRECISION (+)</span><strong>{float(row["Precision (+)"]):.3f}</strong></div>'
        f'<div><span>RECALL (+)</span><strong>{float(row["Recall (+)"]):.3f}</strong></div>'
        f'<div><span>F1 (+)</span><strong>{float(row["F1 (+)"]):.3f}</strong></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def chart_style(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=25, r=20, t=30, b=25),
        paper_bgcolor="#101f34",
        plot_bgcolor="#101f34",
        font=dict(color="#a9c2db", size=12),
        title=None,
        legend_title_text=None,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#314663")
    fig.update_yaxes(gridcolor="#243852", zeroline=False)
    return fig


PAGES = ["Tổng quan", "Khám phá dữ liệu", "Dự đoán một hồ sơ", "Dự đoán từ CSV"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

st.sidebar.markdown('<div class="eyebrow">PHÒNG THÍ NGHIỆM DỮ LIỆU</div>', unsafe_allow_html=True)
st.sidebar.title("◈ Credit Lab")
st.sidebar.caption("Bộ dữ liệu Credit Approval · UCI")
st.sidebar.divider()
page = st.sidebar.radio("ĐIỀU HƯỚNG", PAGES, key="page")
st.sidebar.divider()
st.sidebar.markdown("**Mô hình đang dùng**  \nLogistic Regression")
st.sidebar.caption("Dữ liệu đã ẩn danh. Đây là bài thực hành học máy, không dùng để xét duyệt tín dụng thực tế.")
st.sidebar.link_button("Xem nguồn dữ liệu ↗", "https://archive.ics.uci.edu/dataset/27/credit+approval", width="stretch")

st.markdown(
    '<div class="topline"><div class="brand"><span class="brand-icon">◈</span>'
    '<span>Credit Lab</span></div><div class="status"><span class="status-dot">●</span>'
    'UCI CREDIT APPROVAL · 690 HỒ SƠ</div></div>',
    unsafe_allow_html=True,
)

if page == "Tổng quan":
    logistic_score = comparison.loc[comparison["Mô hình"] == "Logistic Regression", "Accuracy"].iloc[0]
    st.markdown(
        '<div class="hero"><div><div class="eyebrow">PHÂN TÍCH DỮ LIỆU NÂNG CAO · CREDIT APPROVAL</div>'
        '<h1>Từ dữ liệu đến<br>dự đoán.</h1>'
        '<p>Khám phá 690 hồ sơ đã ẩn danh, so sánh hai mô hình học máy và thử dự đoán '
        'trên một hồ sơ hoặc nhiều dòng CSV.</p></div>'
        f'<div class="hero-art"><span>{logistic_score:.0%}</span><small>ACCURACY TRÊN TẬP TEST</small></div></div>',
        unsafe_allow_html=True,
    )

    section("TỔNG QUAN", "Những con số chính", "Dữ liệu UCI gồm 15 thuộc tính A1–A15; A16 là nhãn kết quả.")
    metrics = st.columns(4, gap="medium")
    with metrics[0]:
        stat_card("Tổng hồ sơ", f"{len(df):,}", "Bộ dữ liệu UCI")
    with metrics[1]:
        stat_card("Đặc trưng", str(len(features)), "A1 đến A15")
    with metrics[2]:
        stat_card("Chấp thuận (+)", f'{int((df["A16"] == "+").sum()):,}', "Nhãn mục tiêu A16")
    with metrics[3]:
        stat_card("Accuracy", f"{logistic_score:.1%}", "Logistic Regression · tập test")

    section("ĐÁNH GIÁ", "Hai mô hình, cùng tập kiểm tra", "552 hồ sơ huấn luyện · 138 hồ sơ kiểm tra · chia tập có stratify.")
    model_cols = st.columns(2, gap="medium")
    for col, (_, row) in zip(model_cols, comparison.iterrows()):
        with col:
            model_card(row, row["Mô hình"] == "Logistic Regression")
    st.markdown('<p class="small-note">Chênh lệch nhỏ giữa hai mô hình chưa chứng minh mô hình nào luôn tốt hơn trên dữ liệu mới.</p>', unsafe_allow_html=True)

    section("QUY TRÌNH", "Khám phá, thử nghiệm, tải kết quả", "Bạn có thể bắt đầu từ bất kỳ mục nào trong thanh điều hướng.")
    steps = [
        ("01", "Khám phá dữ liệu", "Xem phân bố nhãn, biểu đồ thuộc tính số và giá trị thiếu."),
        ("02", "Dự đoán một hồ sơ", "Điều chỉnh 15 thuộc tính đã ẩn danh rồi xem kết quả."),
        ("03", "Dự đoán từ CSV", "Tải file mẫu, nhập nhiều dòng và xuất kết quả."),
    ]
    for col, (number, title, detail) in zip(st.columns(3, gap="medium"), steps):
        with col:
            st.markdown(
                f'<div class="step-card"><div class="step-num">BƯỚC {number}</div>'
                f'<strong>{escape(title)}</strong><p>{escape(detail)}</p></div>',
                unsafe_allow_html=True,
            )

elif page == "Khám phá dữ liệu":
    section("KHÁM PHÁ", "Dữ liệu kể câu chuyện gì?", "So sánh hai lớp kết quả và quan sát phân bố của từng thuộc tính số.")
    rejected = int((df["A16"] == "-").sum())
    accepted = int((df["A16"] == "+").sum())
    st.markdown(
        f'<div class="callout"><div class="eyebrow">PHÂN BỐ NHÃN</div>'
        f'<div class="result-label">{rejected} từ chối · {accepted} chấp thuận</div>'
        '<p>A16 là nhãn đã ẩn danh: dấu “−” là từ chối, dấu “+” là chấp thuận.</p></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("#### Kết quả theo số hồ sơ")
        counts = pd.DataFrame({"Kết quả": ["Từ chối (−)", "Chấp thuận (+)"], "Số hồ sơ": [rejected, accepted]})
        fig = px.bar(counts, x="Kết quả", y="Số hồ sơ", color="Kết quả", text="Số hồ sơ",
                     color_discrete_map={"Từ chối (−)": "#5b8dff", "Chấp thuận (+)": "#5bd7c3"})
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_style(fig), width="stretch", theme=None)
    with right:
        selected = st.selectbox("Chọn thuộc tính số để xem", numeric, index=numeric.index("A11"))
        chart_df = df.assign(Ket_qua=df["A16"].map({"-": "Từ chối (−)", "+": "Chấp thuận (+)"}))
        fig = px.box(chart_df, x="Ket_qua", y=selected, color="Ket_qua",
                     color_discrete_map={"Từ chối (−)": "#5b8dff", "Chấp thuận (+)": "#5bd7c3"}, points="outliers")
        fig.update_layout(showlegend=False, xaxis_title="Kết quả")
        st.plotly_chart(chart_style(fig), width="stretch", theme=None)

    section("CHI TIẾT", "Chất lượng và bảng dữ liệu", "Dữ liệu gốc giúp đối chiếu trước khi đưa vào mô hình.")
    with st.expander("Xem số giá trị thiếu theo cột"):
        missing_table = pd.DataFrame({"Cột": df.columns, "Số ô thiếu": df.isna().sum().values})
        st.dataframe(missing_table, hide_index=True, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch", height=360)

elif page == "Dự đoán một hồ sơ":
    section("THỬ NGHIỆM", "Dự đoán một hồ sơ", "Thay đổi các giá trị bên dưới, sau đó bấm Dự đoán. Tên thuộc tính đã được UCI ẩn danh.")
    numeric_defaults = model.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"].statistics_
    categorical_defaults = model.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["imputer"].statistics_
    st.markdown('<div class="callout"><div class="eyebrow">HƯỚNG DẪN</div><p>Giá trị mặc định lấy từ tập huấn luyện. Các mã như A1, A4 không có tên diễn giải công khai; đừng hiểu chúng là thông tin tín dụng cụ thể.</p></div>', unsafe_allow_html=True)

    with st.form("single_record"):
        values = {}
        st.markdown("### 01 / Thuộc tính số")
        cols = st.columns(3, gap="medium")
        for i, col in enumerate(numeric):
            with cols[i % 3]:
                values[col] = st.number_input(col, value=float(numeric_defaults[i]), format="%.2f")
        st.markdown("### 02 / Thuộc tính phân loại")
        cols = st.columns(3, gap="medium")
        for i, col in enumerate(categorical):
            options = categories[col]
            default = options.index(categorical_defaults[i])
            with cols[i % 3]:
                values[col] = st.selectbox(col, options, index=default)
        submitted = st.form_submit_button("Dự đoán hồ sơ →", type="primary", width="stretch")

    if submitted:
        try:
            result = predict_records(pd.DataFrame([values]))
            label = result.iloc[0]["Du_doan"]
            probability = float(result.iloc[0]["Xac_suat_lop_duoc_duyet"])
            heading = "Chấp thuận (+)" if label == "+" else "Từ chối (−)"
            tone = "result-good" if label == "+" else "result-bad"
            st.markdown(
                f'<div class="callout {tone}"><div class="eyebrow">KẾT QUẢ MÔ HÌNH</div>'
                f'<div class="result-label">{heading}</div>'
                f'<p>Xác suất mô hình gán cho lớp chấp thuận (+): <strong>{probability:.1%}</strong></p></div>',
                unsafe_allow_html=True,
            )
            st.progress(probability)
            st.caption("Xác suất là ước lượng của mô hình, không phải mức bảo đảm đúng hoặc quyết định tín dụng thực tế.")
        except ValueError as exc:
            st.error(str(exc))

else:
    section("DỰ ĐOÁN HÀNG LOẠT", "Tải CSV, nhận kết quả", "Dùng cùng pipeline với trang dự đoán một hồ sơ; tối đa 5.000 dòng mỗi lần.")
    st.markdown('<div class="callout"><div class="eyebrow">ĐỊNH DẠNG FILE</div><div class="result-label">15 cột · A1 đến A15</div><p>File CSV cần có hàng tiêu đề. Ô trống hoặc dấu ? sẽ được xử lý bằng quy tắc đã học từ tập huấn luyện.</p></div>', unsafe_allow_html=True)
    st.download_button(
        "↓ Tải CSV mẫu",
        df[features].head(5).to_csv(index=False).encode("utf-8-sig"),
        file_name="credit_input_sample.csv",
        mime="text/csv",
        width="stretch",
    )
    uploaded = st.file_uploader("Chọn CSV để dự đoán", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, na_values=["?"])
            result = predict_records(raw)
            st.success(f"Đã xử lý {len(result):,} hồ sơ.")
            a, b = st.columns(2, gap="medium")
            with a:
                stat_card("Dự đoán chấp thuận (+)", int((result["Du_doan"] == "+").sum()), "Trong file vừa tải lên")
            with b:
                stat_card("Dự đoán từ chối (−)", int((result["Du_doan"] == "-").sum()), "Trong file vừa tải lên")
            section("KẾT QUẢ", "Bảng dự đoán", "Hai cột cuối là nhãn dự đoán và xác suất cho lớp chấp thuận (+).")
            st.dataframe(result, hide_index=True, width="stretch", height=390)
            st.download_button(
                "↓ Tải kết quả dự đoán",
                result.to_csv(index=False).encode("utf-8-sig"),
                file_name="credit_predictions.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        except Exception as exc:
            st.error(f"Chưa xử lý được file: {exc}")

st.markdown(
    '<div class="footer">Nguồn: Quinlan, J. (1987), Credit Approval — UCI Machine Learning Repository · CC BY 4.0. '
    'Ứng dụng minh họa bài học, không dùng để xét duyệt tín dụng thực tế.</div>',
    unsafe_allow_html=True,
)
