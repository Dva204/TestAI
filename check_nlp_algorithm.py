r"""Bộ kiểm tra hồi quy cho thuật toán NLP tiếng Việt.

Chạy nhanh:
    .\.venv\Scripts\python.exe check_nlp_algorithm.py

Xuất báo cáo JSON và trả mã lỗi nếu có ca sai:
    .\.venv\Scripts\python.exe check_nlp_algorithm.py --strict --json nlp_report.json

Mỗi alias trong FEATURE_ALIASES được kiểm tra tự động. CASES bổ sung các ngữ
cảnh thực tế: câu tự nhiên, không dấu, lỗi gõ, phủ định, tuổi, thời gian, mức
độ, yếu tố nguy cơ, câu gây nhầm và cảnh báo cấp cứu.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from nlp_engine import FEATURE_ALIASES, detect_red_flags, extract_features


ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "model_new" / "feature_names_v2.json"


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    expected: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    expected_red_flags: tuple[str, ...] = ()
    structured: dict[str, str] | None = None


CASES = (
    # Câu tự nhiên và nhiều triệu chứng
    Case("noi-cau-tu-nhien", "Tôi sốt cao, rét run và rất mệt mỏi", ("high_fever", "chills", "fatigue")),
    Case("ngoai-vi-tri", "Đau bụng dưới bên phải, ấn bụng đau", ("right_lower_abdominal_pain", "abdominal_tenderness")),
    Case("rang-ham-mat", "Răng bị sâu, đau răng dữ dội và sưng mặt", ("tooth_cavity", "severe_toothache", "facial_jaw_swelling")),
    Case("tai-mui-hong", "Nghẹt mũi, chảy nước mũi và đau họng", ("nasal_congestion", "runny_nose", "sore_throat")),
    Case("nhi-co-tuoi", "Bé 7 tuổi ho khò khè và sốt", ("age_child", "child_patient", "cough", "wheezing", "fever")),
    # Không dấu, chữ hoa và dấu câu
    Case("khong-dau", "TOI BI DAU HONG, NGHET MUI!", ("sore_throat", "nasal_congestion")),
    Case("dau-cau", "Đau đầu... chóng mặt; buồn nôn?", ("headache", "dizziness", "nausea")),
    # Nhiệt độ dạng số
    Case("sot-39-do", "sốt 39 độ", ("fever", "high_fever")),
    Case("sot-40-cap", "sốt 40 độ", ("fever", "high_fever"), expected_red_flags=("Sốt rất cao (≥40°C)",)),
    Case("so-sinh-sot", "Bé 2 tháng tuổi sốt 38 độ", ("age_infant", "child_patient", "fever"), expected_red_flags=("Trẻ dưới 3 tháng sốt ≥38°C",)),
    Case("nhiet-do-thap-phan", "Nhiệt độ 39,5°C", ("fever", "high_fever")),
    Case("sot-38-do", "sốt 38.2 độ C", ("fever",), ("high_fever",)),
    Case("nhiet-do-binh-thuong", "Nhiệt độ 36,8 độ C", (), ("fever", "high_fever")),
    Case("khong-nham-tuoi-thanh-sot", "Bệnh nhân 39 tuổi đau đầu", ("age_adult", "headache"), ("fever", "high_fever")),
    # Lỗi gõ nhỏ được fuzzy matcher hỗ trợ
    Case("loi-go-dau-dau", "đau đầuu", ("headache",)),
    Case("loi-go-chong-mat", "chóng mặtt", ("dizziness",)),
    Case("loi-go-sau-rang", "sâu răngg", ("tooth_cavity",)),
    # Phủ định: phải tạo đặc trưng âm tính và không giữ đặc trưng dương
    Case("phu-dinh-sot", "Tôi không sốt nhưng ho", ("neg_no_fever", "cough"), ("fever",)),
    Case("phu-dinh-kho-tho", "Ho nhưng không khó thở", ("cough", "neg_no_breathing_difficulty"), ("breathing_difficulty",)),
    Case("phu-dinh-non", "Đau bụng, chưa nôn và không tiêu chảy", ("abdominal_pain", "neg_no_vomiting", "neg_no_diarrhea"), ("vomiting", "diarrhea")),
    Case("phu-dinh-dau-hong", "Tôi không đau họng, chỉ đau đầu", ("neg_no_sore_throat", "headache"), ("sore_throat",)),
    # Tuổi và giới tính
    Case("so-sinh", "Em bé 20 ngày tuổi bỏ bú", ("age_newborn", "newborn_patient")),
    Case("nhu-nhi", "Bé trai 8 tháng tuổi sốt", ("age_infant", "child_patient", "sex_male", "fever")),
    Case("vi-thanh-nien", "Nữ 15 tuổi đau đầu", ("age_adolescent", "headache")),
    Case("nguoi-lon", "Nam 35 tuổi chóng mặt", ("age_adult", "dizziness")),
    Case("cao-tuoi", "Phụ nữ 70 tuổi mệt mỏi", ("age_older_adult", "sex_female", "fatigue")),
    # Khởi phát, thời gian, mức độ và nguy cơ
    Case("khoi-phat-dot-ngot", "Đau đầu đột ngột", ("headache", "onset_sudden")),
    Case("khoi-phat-tu-tu", "Đau tăng dần", ("onset_gradual",)),
    Case("sau-chan-thuong", "Đau sau tai nạn", ("onset_after_trauma",)),
    Case("thoi-gian-gio", "Khó thở được vài giờ", ("breathing_difficulty", "duration_hours")),
    Case("thoi-gian-ngay", "Ho đã 3 ngày", ("cough", "duration_1_3_days")),
    Case("thoi-gian-dai", "Ho lâu ngày hơn một tháng", ("chronic_cough", "duration_over_4_weeks")),
    Case("muc-do-nang", "Đau bụng dữ dội", ("abdominal_pain", "severity_severe"), expected_red_flags=("Đau bụng cấp",)),
    Case("hut-thuoc", "Người bệnh hút thuốc và ho kéo dài", ("risk_smoking", "chronic_cough")),
    Case("tieu-duong", "Có tiền sử đái tháo đường", ("risk_diabetes",)),
    Case("mang-thai", "Phụ nữ có bầu bị buồn nôn", ("sex_female", "pregnancy_possible", "nausea")),
    Case("ve-sinh-rang", "Ít đánh răng, hơi thở hôi", ("risk_poor_oral_hygiene", "bad_breath")),
    # Những câu từng dễ ánh xạ nhầm
    Case("khong-nham-dau-dau-dau-hong", "đau đầu", ("headache",), ("sore_throat", "facial_pain_pressure")),
    Case("khong-nham-mat-thang-bang", "đau đầu, chóng mặt, mất thăng bằng", ("headache", "dizziness", "imbalance"), ("sore_throat", "facial_pain_pressure")),
    Case("tu-ngoai-ngu-canh", "Tôi cần đặt lịch khám vào ngày mai", (), ("fever", "headache", "cough")),
    Case("noi-ve-nguoi-khac", "Mẹ tôi nói thời tiết hôm nay nóng", (), ("fever",)),
    # Cảnh báo cấp cứu
    Case("co-giat", "Bệnh nhân lên cơn giật", expected_red_flags=("Co giật",)),
    Case("dot-quy", "Đột ngột méo miệng và liệt nửa người", expected_red_flags=("Dấu hiệu đột quỵ",)),
    Case("xuat-huyet", "Nôn ra máu", ("vomiting",), expected_red_flags=("Xuất huyết tiêu hóa",)),
    Case("mat-thinh-luc", "Mất thính lực đột ngột", expected_red_flags=("Mất thính lực đột ngột",)),
    # Dữ liệu từ các ô lựa chọn trên giao diện
    Case("structured", "đau đầu", ("headache", "age_adult", "severity_moderate"), structured={"age": "age_adult", "severity": "severity_moderate"}),
)


def load_feature_names() -> list[str]:
    with FEATURES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def all_cases(feature_names: list[str]) -> list[Case]:
    cases = list(CASES)
    valid = set(feature_names)
    # Đây là lớp kiểm tra độ phủ lớn nhất: mọi cách gọi đã khai báo đều phải
    # quay về đúng feature và không được hỏng sau mỗi lần sửa thuật toán.
    for feature, aliases in FEATURE_ALIASES.items():
        if feature not in valid:
            continue
        for index, alias in enumerate(aliases, start=1):
            cases.append(Case(f"alias:{feature}:{index}", alias, (feature,)))
    return cases


def evaluate(case: Case, feature_names: list[str]) -> dict:
    result = extract_features(case.text, feature_names, case.structured)
    active = set(result["active_features"])
    red_flags = set(detect_red_flags(case.text))
    missing = sorted(set(case.expected) - active)
    forbidden_found = sorted(set(case.forbidden) & active)
    missing_red_flags = sorted(set(case.expected_red_flags) - red_flags)
    passed = not (missing or forbidden_found or missing_red_flags)
    return {
        "name": case.name,
        "text": case.text,
        "passed": passed,
        "expected": list(case.expected),
        "active": sorted(active),
        "missing": missing,
        "forbidden_found": forbidden_found,
        "expected_red_flags": list(case.expected_red_flags),
        "detected_red_flags": sorted(red_flags),
        "missing_red_flags": missing_red_flags,
        "detections": [list(item) for item in result["detections"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra hồi quy NLP y khoa tiếng Việt")
    parser.add_argument("--json", type=Path, help="Đường dẫn lưu báo cáo JSON")
    parser.add_argument("--strict", action="store_true", help="Trả mã lỗi 1 nếu có ca kiểm tra sai")
    parser.add_argument("--show-passed", action="store_true", help="Hiện cả các ca đã đạt")
    args = parser.parse_args()

    feature_names = load_feature_names()
    results = [evaluate(case, feature_names) for case in all_cases(feature_names)]
    failures = [result for result in results if not result["passed"]]
    orphan_aliases = sorted(set(FEATURE_ALIASES) - set(feature_names))

    for result in results:
        if result["passed"] and not args.show_passed:
            continue
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']}: {result['text']}")
        if not result["passed"]:
            print(f"  thiếu: {result['missing'] or '-'}")
            print(f"  nhận sai: {result['forbidden_found'] or '-'}")
            print(f"  thiếu cảnh báo: {result['missing_red_flags'] or '-'}")
            print(f"  active: {result['active']}")

    passed = len(results) - len(failures)
    print(f"\nKết quả: {passed}/{len(results)} đạt ({passed / len(results) * 100:.1f}%)")
    print(f"Số ca cần tối ưu: {len(failures)}")
    print(f"Alias không có cột tương ứng trong model: {len(orphan_aliases)}")
    if orphan_aliases:
        print("  " + ", ".join(orphan_aliases))

    if args.json:
        report = {
            "summary": {"total": len(results), "passed": passed, "failed": len(failures)},
            "orphan_aliases": orphan_aliases,
            "cases": results,
            "case_definitions": [asdict(case) for case in CASES],
        }
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu báo cáo: {args.json.resolve()}")

    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
