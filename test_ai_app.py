from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from nlp_engine import FEATURE_ALIASES, detect_red_flags, extract_features

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_new"

st.set_page_config(page_title="Gợi ý chuyên khoa AI", page_icon="🏥", layout="wide")


@st.cache_resource
def load_resources():
    bundle = joblib.load(MODEL_DIR / "ai_disease_model_v2.joblib")
    required = {"model", "feature_names", "disease_map"}
    if missing := required - set(bundle):
        raise ValueError(f"Model bundle thiếu khóa: {sorted(missing)}")
    features = list(bundle["feature_names"])
    if len(features) != 450:
        raise ValueError(f"Cần 450 đặc trưng, nhận được {len(features)}")
    mapping_path = MODEL_DIR / "disease_specialty_map_v2.csv"
    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    return bundle, bundle["model"], features, mapping


@st.cache_data
def load_feature_specificity(feature_names_tuple):
    """Estimate how strongly each feature points to one specialty in training data."""
    training_path = ROOT / "training_data_5_specialties_v2.csv"
    mapping_path = MODEL_DIR / "disease_specialty_map_v2.csv"
    if not training_path.exists() or not mapping_path.exists():
        return {}

    training = pd.read_csv(training_path, encoding="utf-8-sig")
    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    disease_to_specialty = mapping.set_index("disease_key")["specialty"].to_dict()
    row_specialties = training["prognosis"].map(disease_to_specialty)
    specialties = sorted(row_specialties.dropna().unique())
    available_features = [name for name in feature_names_tuple if name in training.columns]
    if not available_features or not specialties:
        return {}

    feature_matrix = training[available_features].fillna(0).to_numpy(dtype=float)
    specialty_matrix = np.column_stack([
        (row_specialties == specialty).to_numpy(dtype=float)
        for specialty in specialties
    ])
    counts = feature_matrix.T @ specialty_matrix
    specificity = {}
    for feature_index, feature in enumerate(available_features):
        support = int(counts[feature_index].sum())
        if support == 0:
            continue
        best_index = int(np.argmax(counts[feature_index]))
        specificity[feature] = {
            "specialty": specialties[best_index],
            "score": float(counts[feature_index, best_index] / support),
            "support": support,
        }
    return specificity


try:
    bundle, model, feature_names, disease_map = load_resources()
except Exception as exc:
    st.error(f"Không thể nạp model mới: {exc}")
    st.info("Cài thư viện bằng `python -m pip install -r requirements.txt` rồi chạy lại.")
    st.stop()

mapping_lookup = disease_map.set_index("disease_key").to_dict("index")
feature_specificity = load_feature_specificity(tuple(feature_names))


def predict(vector):
    frame = pd.DataFrame(
        [[vector[name] for name in feature_names]],
        columns=feature_names,
        dtype=np.int8,
    )
    probabilities = model.predict_proba(frame)[0]
    probabilities = probabilities.astype(float, copy=True)

    pediatric_evidence_features = {
        "age_newborn",
        "age_infant",
        "age_child",
        "age_adolescent",
        "child_patient",
        "newborn_patient",
    }
    has_pediatric_evidence = any(
        vector.get(feature, 0) == 1
        for feature in pediatric_evidence_features
    )

    # Không cho model tự suy đoán bệnh nhi chỉ từ triệu chứng chung như ho/sốt.
    # Khi người dùng không nói là trẻ em, loại các lớp thuộc khoa Nhi rồi chuẩn
    # hóa lại xác suất của các bệnh còn đủ điều kiện.
    if not has_pediatric_evidence:
        for index, disease_class in enumerate(model.classes_):
            info = mapping_lookup.get(str(disease_class))
            if info and info["specialty"] == "Nhi":
                probabilities[index] = 0.0
        remaining_probability = probabilities.sum()
        if remaining_probability > 0:
            probabilities /= remaining_probability
    disease_predictions = []
    specialty_scores = {
        specialty: 0.0
        for specialty in sorted(disease_map["specialty"].dropna().unique())
    }

    # P(chuyên khoa | triệu chứng) = tổng xác suất của mọi bệnh thuộc khoa đó.
    for index, probability in enumerate(probabilities):
        disease_key = str(model.classes_[index])
        info = mapping_lookup.get(disease_key)
        if info and info["specialty"] in specialty_scores:
            specialty_scores[info["specialty"]] += float(probability)

    for index in np.argsort(probabilities)[::-1][:5]:
        key = str(model.classes_[index])
        info = mapping_lookup.get(key, {"disease_name_vi": key, "specialty": "Nội tổng quát"})
        disease_predictions.append({
            "key": key,
            "name": info["disease_name_vi"],
            "specialty": info["specialty"],
            "probability": float(probabilities[index]),
        })
    specialty_predictions = [
        {"specialty": specialty, "probability": probability}
        for specialty, probability in sorted(
            specialty_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return {
        "diseases": disease_predictions,
        "specialties": specialty_predictions,
        "pediatric_evidence": has_pediatric_evidence,
    }


def show_result(result, extraction):
    disease_predictions = result["diseases"]
    specialty_predictions = result["specialties"]
    first_disease = disease_predictions[0]
    first_specialty = specialty_predictions[0]
    disease_margin = first_disease["probability"] - disease_predictions[1]["probability"]
    specialty_margin = first_specialty["probability"] - specialty_predictions[1]["probability"]
    uncertain_routing = (
        first_specialty["probability"] < 0.50
        or specialty_margin < 0.10
    )
    routed_specialty = (
        "Nội tổng quát"
        if uncertain_routing
        else first_specialty["specialty"]
    )
    positive_features = {
        feature
        for feature, _label, method, _score in extraction["detections"]
        if method != "negative"
    }
    specific_evidence = sorted(
        (
            (feature, feature_specificity[feature])
            for feature in positive_features
            if feature in feature_specificity
            and feature_specificity[feature]["score"] >= 0.80
            and feature_specificity[feature]["support"] >= 20
        ),
        key=lambda item: (item[1]["score"], item[1]["support"]),
        reverse=True,
    )
    evidence_is_sufficient = (
        extraction["clinical_match_count"] >= 2
        or bool(specific_evidence)
    )
    if not evidence_is_sufficient:
        st.warning(
            "Dấu hiệu hiện tại còn chung hoặc chưa đủ đặc hiệu để định tuyến an toàn. "
            "Hãy bổ sung vị trí, thời gian, mức độ hoặc 1–2 dấu hiệu đi kèm."
        )
        st.metric("Khoa tiếp nhận ban đầu", "Nội tổng quát")
        return

    if extraction["clinical_match_count"] == 1 and specific_evidence:
        feature, evidence = specific_evidence[0]
        st.info(
            f"Một dấu hiệu vẫn được chấp nhận vì `{feature}` có tính định hướng cao tới "
            f"**{evidence['specialty']}** trong dữ liệu huấn luyện "
            f"({evidence['score'] * 100:.0f}% trên {evidence['support']} mẫu có dấu hiệu này)."
        )

    if not result["pediatric_evidence"]:
        st.caption(
            "Khoa Nhi không được tính vào xếp hạng vì chưa có thông tin cho thấy "
            "người bệnh là trẻ em. Hãy chọn nhóm tuổi hoặc ghi tuổi trong mô tả nếu cần."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bệnh gợi ý", first_disease["name"])
    c2.metric("Độ tin cậy bệnh", f"{first_disease['probability'] * 100:.1f}%")
    c3.metric("Khoa nên khám", routed_specialty)
    c4.metric(
        "Độ tin cậy chuyên khoa",
        "Chưa đủ" if uncertain_routing else f"{first_specialty['probability'] * 100:.1f}%",
    )

    if uncertain_routing:
        st.warning(
            f"Khoa có điểm mô hình cao nhất là **{first_specialty['specialty']}** "
            f"({first_specialty['probability'] * 100:.1f}%), nhưng chưa đạt ngưỡng định tuyến. "
            "Hệ thống chuyển về **Nội tổng quát** để tiếp nhận ban đầu."
        )
    else:
        st.success(
            f"Khoa được gợi ý cao nhất: **{first_specialty['specialty']}** "
            f"({first_specialty['probability'] * 100:.1f}%)."
        )

    if first_disease["probability"] < 0.25 or disease_margin < 0.05:
        st.info(
            "Tên bệnh cụ thể có độ chắc chắn thấp hơn gợi ý chuyên khoa. "
            "Đây là điều bình thường khi nhiều bệnh cùng khoa có triệu chứng tương tự."
        )

    with st.expander("Các dấu hiệu NLP đã nhận diện", expanded=True):
        if not extraction["detections"]:
            st.write("Chưa tìm thấy cụm triệu chứng trong kho từ vựng.")
        for feature, label, method, score in extraction["detections"]:
            method_label = {
                "exact": "khớp trực tiếp",
                "fuzzy": f"sửa lỗi gần đúng {score * 100:.0f}%",
                "negative": "phủ định rõ ràng",
                "numeric": "đọc nhiệt độ dạng số",
            }.get(method, f"phương thức {method}")
            st.write(f"- **{label}** → `{feature}` ({method_label})")

    with st.expander("Độ tin cậy theo 5 chuyên khoa", expanded=True):
        for item in specialty_predictions:
            st.write(
                f"**{item['specialty']}** — {item['probability'] * 100:.2f}%"
            )
            st.progress(min(item["probability"], 1.0))

    with st.expander("5 bệnh có khả năng cao nhất"):
        for item in disease_predictions:
            st.write(
                f"**{item['name']}** — {item['specialty']} — "
                f"{item['probability'] * 100:.2f}%"
            )
            st.progress(min(item["probability"], 1.0))


st.title("Hệ thống NLP gợi ý bệnh và chuyên khoa")
st.caption(
    "Model v2: 100 bệnh, 450 đặc trưng, 5 chuyên khoa. "
    "Kết quả hỗ trợ chọn khoa, không thay thế chẩn đoán của bác sĩ."
)

tab_nlp, tab_checklist, tab_model = st.tabs(
    ["Nhập mô tả tự nhiên", "Chọn triệu chứng", "Thông tin model"]
)

with tab_nlp:
    st.subheader("Mô tả triệu chứng bằng tiếng Việt")
    st.caption(
        "Hỗ trợ câu có dấu/không dấu, phủ định và lỗi gõ gần đúng. "
        "Nên ghi tuổi, thời gian, vị trí và mức độ."
    )
    text = st.text_area(
        "Triệu chứng",
        placeholder="Ví dụ: Bé trai 6 tuổi sốt cao 2 ngày, ho, thở nhanh và rút lõm lồng ngực.",
        height=140,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    age = c1.selectbox("Nhóm tuổi", ["", "age_newborn", "age_infant", "age_child", "age_adolescent", "age_adult", "age_older_adult"], format_func=lambda x: {"": "Tự nhận diện", "age_newborn": "Sơ sinh", "age_infant": "Nhũ nhi", "age_child": "Trẻ em", "age_adolescent": "Vị thành niên", "age_adult": "Người lớn", "age_older_adult": "Cao tuổi"}[x])
    sex = c2.selectbox("Giới tính", ["", "sex_female", "sex_male"], format_func=lambda x: {"": "Không khai báo", "sex_female": "Nữ", "sex_male": "Nam"}[x])
    duration = c3.selectbox("Thời gian", ["", "duration_hours", "duration_1_3_days", "duration_4_7_days", "duration_1_4_weeks", "duration_over_4_weeks"], format_func=lambda x: {"": "Tự nhận diện", "duration_hours": "Vài giờ", "duration_1_3_days": "1–3 ngày", "duration_4_7_days": "4–7 ngày", "duration_1_4_weeks": "1–4 tuần", "duration_over_4_weeks": "Trên 4 tuần"}[x])
    onset = c4.selectbox("Khởi phát", ["", "onset_sudden", "onset_gradual", "onset_after_meal", "onset_after_trauma", "onset_after_infection"], format_func=lambda x: {"": "Tự nhận diện", "onset_sudden": "Đột ngột", "onset_gradual": "Từ từ", "onset_after_meal": "Sau ăn", "onset_after_trauma": "Sau chấn thương", "onset_after_infection": "Sau nhiễm bệnh"}[x])
    severity = c5.selectbox("Mức độ", ["", "severity_mild", "severity_moderate", "severity_severe"], format_func=lambda x: {"": "Tự nhận diện", "severity_mild": "Nhẹ", "severity_moderate": "Vừa", "severity_severe": "Nặng"}[x])

    if st.button("Phân tích NLP và dự đoán", type="primary"):
        if not text.strip():
            st.warning("Vui lòng nhập mô tả triệu chứng.")
        else:
            if red_flags := detect_red_flags(text):
                temperature_flags = {
                    "Sốt rất cao (≥40°C)",
                    "Trẻ dưới 3 tháng sốt ≥38°C",
                }
                if set(red_flags).issubset(temperature_flags):
                    st.error(
                        "Cần được đánh giá y tế khẩn: " + ", ".join(red_flags)
                        + ". Hãy đến cơ sở y tế/cấp cứu ngay. Gọi 115 nếu người bệnh "
                        "khó thở, co giật, lơ mơ, tím tái hoặc không thể tự di chuyển an toàn."
                    )
                else:
                    st.error(
                        "Dấu hiệu cần đánh giá khẩn cấp: " + ", ".join(red_flags)
                        + ". Hãy gọi 115 hoặc đến cơ sở cấp cứu gần nhất."
                    )
            structured = {"age": age, "sex": sex, "duration": duration, "onset": onset, "severity": severity}
            extraction = extract_features(text, feature_names, structured)
            show_result(predict(extraction["vector"]), extraction)

with tab_checklist:
    st.subheader("Chọn triệu chứng đã chuẩn hóa")
    options = {
        f"{aliases[0]} ({feature})": feature
        for feature, aliases in FEATURE_ALIASES.items()
        if feature in feature_names
    }
    selected = st.multiselect(
        "Triệu chứng",
        sorted(options),
        placeholder="Gõ: đau bụng, khó thở, đau răng, đau tai...",
    )
    if st.button("Dự đoán từ checklist", type="primary"):
        selected_features = [options[label] for label in selected]
        if len(selected_features) < 2:
            st.warning("Hãy chọn ít nhất 2 triệu chứng.")
        else:
            vector = {name: int(name in selected_features) for name in feature_names}
            extraction = {
                "clinical_match_count": len(selected_features),
                "detections": [(f, FEATURE_ALIASES[f][0], "exact", 1.0) for f in selected_features],
            }
            show_result(predict(vector), extraction)

with tab_model:
    st.subheader("Thông tin model đang sử dụng")
    metrics = bundle.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số bệnh", bundle.get("dataset_classes", len(disease_map)))
    c2.metric("Đặc trưng", len(feature_names))
    c3.metric("Số mẫu", bundle.get("dataset_rows", "Không rõ"))
    c4.metric("Macro F1 Kaggle", f"{metrics.get('macro_f1', 0) * 100:.2f}%")
    st.write("**Chuyên khoa:** " + ", ".join(sorted(disease_map["specialty"].unique())))
    st.dataframe(
        disease_map[["disease_name_vi", "specialty"]].sort_values(["specialty", "disease_name_vi"]),
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "Model được train bằng dữ liệu tổng hợp. Điểm cao trên Kaggle chưa chứng minh "
        "độ chính xác trên bệnh nhân thật; cần bác sĩ thẩm định trước khi triển khai."
    )
