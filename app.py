from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Credit Lab",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
div[data-testid="stMetric"] * {
    color: #17243b !important;
}
<style>
.stApp {
    background: #f6f8fc;
}
.block-container {
    max-width: 1080px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
section[data-testid="stSidebar"] {
    border-right: 0;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e1e7f0;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 8px 24px rgba(23, 36, 59, .06);
}
div[data-testid="stMetric"] * {
    color: #17243b !important;
}
div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e1e7f0;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(23, 36, 59, .05);
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div {
    border-radius: 10px;
}
.hero {
    background: linear-gradient(115deg, #172554, #4338ca);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 30px;
    box-shadow: 0 14px 32px rgba(44, 46, 142, .16);
}
.hero h1 {
    color: #ffffff;
    font-size: 2.7rem;
    margin: 0;
}
.hero p {
    color: #e0e7ff;
    font-size: 1.05rem;
    margin: 8px 0 0;
}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE / "credit_model.joblib")

@st.cache_data
def load_tables():
    return (
        pd.read_csv(BASE / "credit_data.csv"),
        pd.read_csv(BASE / "model_comparison.csv")
    )

try:
    artifacts = load_model()
    df, comparison = load_tables()
except Exception as exc:
    st.error(
        "Chưa tải được model hoặc dữ liệu. Kiểm tra các file "
        "credit_model.joblib, credit_data.csv và model_comparison.csv."
    )
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

categories = {
    col: list(values)
    for col, values in zip(categorical, encoder.categories_)
}

positive_index = list(model.classes_).index("+")

def predict_records(raw):
    """Kiểm tra đầu vào và dự đoán bằng pipeline đã lưu."""
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
    data = data.replace(r"^\s*$", np.nan, regex=True)
    data = data.replace("?", np.nan)

    for col in numeric:
        converted = pd.to_numeric(data[col], errors="coerce")
        invalid = data[col].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Cột {col} có giá trị không phải số.")
        if np.isinf(converted.to_numpy(dtype=float)).any():
            raise ValueError(f"Cột {col} có giá trị vô hạn.")
        data[col] = converted

    for col in categorical:
        data[col] = data[col].map(
            lambda value: str(value).strip()
            if pd.notna(value) else np.nan
        )
        invalid = data[col].notna() & ~data[col].isin(categories[col])
        if invalid.any():
            raise ValueError(
                f"Cột {col} có mã chưa được hỗ trợ. "
                f"Các mã hợp lệ: {', '.join(map(str, categories[col]))}."
            )

    result = data.copy()
    result["Du_doan"] = model.predict(data)
    result["Xac_suat_lop_duoc_duyet"] = (
        model.predict_proba(data)[:, positive_index]
    )
    return result

st.sidebar.title("📊 Credit Lab")
st.sidebar.caption("PHÂN TÍCH DỮ LIỆU NÂNG CAO")
page = st.sidebar.radio(
    "Đi đến",
    ["Tổng quan", "Khám phá dữ liệu",
     "Dự đoán một hồ sơ", "Dự đoán từ CSV"]
)
st.sidebar.divider()
st.sidebar.write("**Mô hình:** Logistic Regression")
st.sidebar.caption(
    "Bài thực hành trên dữ liệu UCI đã ẩn danh. "
    "Kết quả dùng để minh họa học máy, không dùng xét duyệt tín dụng thực tế."
)
st.sidebar.markdown(
    "[Nguồn dữ liệu · UCI](https://archive.ics.uci.edu/dataset/27/credit+approval)"
)

st.markdown("""
<div class="hero">
    <h1>Credit Lab</h1>
    <p>Khám phá dữ liệu · Đánh giá mô hình · Thử nghiệm dự đoán</p>
</div>
""", unsafe_allow_html=True)

if page == "Tổng quan":
    st.subheader("Từ dữ liệu đến dự đoán")
    st.write(
        "Bộ dữ liệu gồm các thuộc tính A1–A15 và nhãn A16. "
        "UCI đã ẩn danh ý nghĩa thuộc tính; + là chấp thuận, - là từ chối."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hồ sơ", f"{len(df):,}")
    c2.metric("Đặc trưng", len(features))
    c3.metric("Nhãn chấp thuận", int((df["A16"] == "+").sum()))

    score = comparison.loc[
        comparison["Mô hình"] == "Logistic Regression", "Accuracy"
    ].iloc[0]
    c4.metric("Accuracy trên tập test", f"{score:.1%}")

    st.subheader("So sánh mô hình trên cùng tập test")
    st.dataframe(
        comparison.style.format({
            col: "{:.3f}" for col in comparison.columns
            if col != "Mô hình"
        }),
        hide_index=True,
        width="stretch"
    )
    st.caption(
        "552 hồ sơ huấn luyện, 138 hồ sơ kiểm tra. "
        "Chênh lệch nhỏ giữa hai mô hình chưa chứng minh một mô hình "
        "luôn tốt hơn trên dữ liệu mới."
    )

    st.subheader("Cách sử dụng")
    st.write(
        "① Khám phá phân phối dữ liệu. "
        "② Nhập các thuộc tính của một hồ sơ. "
        "③ Hoặc tải CSV để xem và tải về dự đoán hàng loạt."
    )

elif page == "Khám phá dữ liệu":
    left, right = st.columns(2)

    counts = df["A16"].value_counts().rename_axis(
        "Kết quả"
    ).reset_index(name="Số hồ sơ")

    with left:
        st.plotly_chart(
            px.bar(
                counts, x="Kết quả", y="Số hồ sơ",
                color="Kết quả",
                color_discrete_map={"-": "#6366f1", "+": "#14b8a6"},
                title="Phân bố kết quả", text_auto=True
            ),
            width="stretch"
        )

    with right:
        selected = st.selectbox(
            "Chọn thuộc tính số", numeric, index=numeric.index("A11")
        )
        st.plotly_chart(
            px.box(
                df, x="A16", y=selected, color="A16",
                color_discrete_map={"-": "#6366f1", "+": "#14b8a6"},
                title=f"Phân bố {selected} theo kết quả"
            ),
            width="stretch"
        )

    with st.expander("Kiểm tra giá trị thiếu"):
        missing_table = pd.DataFrame({
            "Cột": df.columns,
            "Số ô thiếu": df.isna().sum().values
        })
        st.dataframe(missing_table, hide_index=True, width="stretch")

    st.subheader("Dữ liệu gốc")
    st.dataframe(df, hide_index=True, width="stretch")

elif page == "Dự đoán một hồ sơ":
    st.subheader("Thử nghiệm một hồ sơ")
    st.caption(
        "Các giá trị ban đầu là trung vị và mã phổ biến trong tập train. "
        "Bạn có thể thay đổi chúng rồi bấm Dự đoán."
    )

    numeric_defaults = (
        model.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"].statistics_
    )
    categorical_defaults = (
        model.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["imputer"].statistics_
    )

    with st.form("single_record"):
        values = {}
        st.markdown("**Thuộc tính số**")
        cols = st.columns(3)
        for i, col in enumerate(numeric):
            with cols[i % 3]:
                values[col] = st.number_input(
                    col, value=float(numeric_defaults[i]),
                    format="%.2f"
                )

        st.markdown("**Thuộc tính phân loại**")
        cols = st.columns(3)
        for i, col in enumerate(categorical):
            options = categories[col]
            default = options.index(categorical_defaults[i])
            with cols[i % 3]:
                values[col] = st.selectbox(
                    col, options, index=default
                )

        submitted = st.form_submit_button(
            "Dự đoán hồ sơ", type="primary"
        )

    if submitted:
        try:
            result = predict_records(pd.DataFrame([values]))
            label = result.iloc[0]["Du_doan"]
            probability = float(
                result.iloc[0]["Xac_suat_lop_duoc_duyet"]
            )

            a, b = st.columns(2)
            a.metric(
                "Nhãn mô hình dự đoán",
                "Chấp thuận (+)" if label == "+" else "Từ chối (-)"
            )
            b.metric("Xác suất mô hình cho lớp +", f"{probability:.1%}")
            st.progress(probability)
            st.caption(
                "Xác suất do mô hình ước lượng, không phải mức bảo đảm đúng."
            )
        except ValueError as exc:
            st.error(str(exc))

else:
    st.subheader("Dự đoán hàng loạt bằng CSV")
    st.write(
        "CSV cần đủ 15 cột A1–A15. Ô trống hoặc dấu ? được xử lý "
        "bằng quy tắc đã học từ tập train."
    )

    st.download_button(
        "Tải CSV mẫu",
        df[features].head(5).to_csv(index=False).encode("utf-8-sig"),
        file_name="credit_input_sample.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, na_values=["?"])
            result = predict_records(raw)

            st.success(f"Đã dự đoán {len(result):,} hồ sơ.")
            st.dataframe(result, hide_index=True, width="stretch")
            st.download_button(
                "Tải kết quả dự đoán",
                result.to_csv(index=False).encode("utf-8-sig"),
                file_name="credit_predictions.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Chưa xử lý được file: {exc}")

st.divider()
st.caption(
    "Nguồn: Quinlan, J. (1987), Credit Approval — "
    "UCI Machine Learning Repository · CC BY 4.0."
)
