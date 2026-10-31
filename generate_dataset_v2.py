"""Generate an expanded, auditable synthetic dataset for five specialties.

Educational/prototyping use only. The profiles are not a validated clinical
dataset and require specialist review before any real-world use.
"""
from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from generate_5_specialty_dataset import PROFILES as BASE_PROFILES

ROOT = Path(__file__).resolve().parent
SEED = 20261024
SAMPLES_PER_DISEASE = 100


def p(key, vi, specialty, symptoms):
    symptoms = symptoms.split()
    return (key, vi, specialty, symptoms[:2], symptoms[2:])


# Fifteen additional diseases per specialty. Together with the original five,
# this gives 100 diseases (20 per specialty).
ADDITIONAL = [
    # Nội tổng quát
    p("community_acquired_pneumonia", "Viêm phổi cộng đồng", "Nội tổng quát", "cough fever productive_cough breathing_difficulty chest_pain chills crackles low_oxygen"),
    p("bronchial_asthma", "Hen phế quản", "Nội tổng quát", "wheezing episodic_breathlessness chest_tightness cough nighttime_symptoms allergen_trigger prolonged_expiration"),
    p("copd_exacerbation", "Đợt cấp COPD", "Nội tổng quát", "chronic_cough worsening_breathlessness increased_sputum purulent_sputum wheezing smoking_history low_oxygen"),
    p("influenza", "Cúm", "Nội tổng quát", "acute_fever myalgia dry_cough headache fatigue sore_throat chills"),
    p("dengue", "Sốt xuất huyết Dengue", "Nội tổng quát", "high_fever severe_myalgia headache pain_behind_eyes rash bleeding_gums abdominal_pain warning_lethargy"),
    p("iron_deficiency_anemia", "Thiếu máu thiếu sắt", "Nội tổng quát", "fatigue pallor exertional_breathlessness dizziness palpitations brittle_nails pica heavy_menses"),
    p("hyperthyroidism", "Cường giáp", "Nội tổng quát", "weight_loss palpitations heat_intolerance tremor sweating anxiety increased_appetite goiter"),
    p("hypothyroidism", "Suy giáp", "Nội tổng quát", "fatigue cold_intolerance weight_gain constipation dry_skin slow_heart_rate puffy_face hoarse_voice"),
    p("gout", "Gút", "Nội tổng quát", "acute_joint_pain red_swollen_joint big_toe_joint_pain severe_tenderness recurrent_attacks high_purine_risk"),
    p("rheumatoid_arthritis", "Viêm khớp dạng thấp", "Nội tổng quát", "symmetric_joint_pain morning_stiffness swollen_small_joints fatigue reduced_grip joint_warmth"),
    p("chronic_kidney_disease", "Bệnh thận mạn", "Nội tổng quát", "leg_swelling fatigue reduced_urine nausea itchy_skin high_blood_pressure nocturia pallor"),
    p("urinary_tract_infection", "Nhiễm trùng đường tiết niệu", "Nội tổng quát", "painful_urination frequent_urination urinary_urgency lower_abdominal_pain cloudy_urine foul_smelling_urine fever"),
    p("hepatitis", "Viêm gan", "Nội tổng quát", "jaundice dark_urine fatigue loss_of_appetite nausea right_upper_abdominal_pain pale_stool"),
    p("peptic_ulcer", "Loét dạ dày tá tràng", "Nội tổng quát", "epigastric_pain meal_related_pain nausea bloating black_stool vomiting_blood nsaid_risk"),
    p("heart_failure", "Suy tim", "Nội tổng quát", "exertional_breathlessness leg_swelling orthopnea nighttime_breathlessness fatigue lung_crackles rapid_heart_rate"),

    # Ngoại
    p("bowel_obstruction", "Tắc ruột", "Ngoại", "colicky_abdominal_pain vomiting abdominal_distension no_stool_or_gas high_pitched_bowel_sounds dehydration prior_surgery_risk"),
    p("perforated_peptic_ulcer", "Thủng ổ loét dạ dày tá tràng", "Ngoại", "sudden_severe_abdominal_pain board_like_abdomen abdominal_guarding nausea shock_signs free_air_sign"),
    p("acute_pancreatitis", "Viêm tụy cấp", "Ngoại", "severe_epigastric_pain pain_to_back vomiting abdominal_tenderness alcohol_risk gallstone_risk fever"),
    p("diverticulitis", "Viêm túi thừa đại tràng", "Ngoại", "left_lower_abdominal_pain fever bowel_habit_change abdominal_tenderness nausea older_age"),
    p("perianal_abscess", "Áp xe hậu môn", "Ngoại", "severe_anal_pain perianal_swelling fever pus_discharge pain_when_sitting diabetes_risk"),
    p("anal_fissure", "Nứt kẽ hậu môn", "Ngoại", "sharp_pain_with_defecation bright_rectal_bleeding constipation anal_spasm fear_of_defecation"),
    p("pilonidal_sinus", "Nang lông vùng cùng cụt", "Ngoại", "painful_tailbone_lump pus_discharge skin_redness fever prolonged_sitting_risk recurrent_swelling"),
    p("varicose_veins", "Suy giãn tĩnh mạch", "Ngoại", "visible_leg_veins leg_heaviness leg_aching ankle_swelling worse_after_standing skin_discoloration"),
    p("diabetic_foot_infection", "Nhiễm trùng bàn chân đái tháo đường", "Ngoại", "foot_ulcer pus_discharge skin_redness foot_swelling fever diabetes_history reduced_foot_sensation"),
    p("breast_abscess", "Áp xe vú", "Ngoại", "painful_breast_lump breast_redness fever breast_warmth pus_discharge breastfeeding_risk"),
    p("thyroid_nodule", "Nhân tuyến giáp", "Ngoại", "neck_lump thyroid_nodule difficulty_swallowing hoarse_voice rapid_growth_sign enlarged_neck_nodes"),
    p("soft_tissue_tumor", "U mô mềm", "Ngoại", "growing_soft_tissue_lump painless_lump firm_mass fixed_mass large_mass_size unexplained_weight_loss"),
    p("lipoma", "U mỡ", "Ngoại", "soft_subcutaneous_lump mobile_lump painless_lump slow_growth normal_overlying_skin"),
    p("ganglion_cyst", "Nang hoạt dịch", "Ngoại", "wrist_hand_lump size_changes_with_activity joint_discomfort firm_cyst reduced_joint_motion"),
    p("traumatic_wound", "Vết thương phần mềm", "Ngoại", "open_wound active_bleeding wound_pain contaminated_wound foreign_body_sign numbness_near_wound"),

    # Nhi
    p("pediatric_pneumonia", "Viêm phổi trẻ em", "Nhi", "child_patient cough fever rapid_breathing chest_indrawing low_oxygen poor_feeding lethargy"),
    p("acute_diarrhea_child", "Tiêu chảy cấp trẻ em", "Nhi", "child_patient diarrhea vomiting dehydration poor_drinking sunken_eyes reduced_urine blood_in_stool"),
    p("measles", "Sởi", "Nhi", "child_patient fever generalized_rash cough runny_nose red_eyes koplik_spots"),
    p("chickenpox", "Thủy đậu", "Nhi", "child_patient itchy_vesicular_rash fever rash_in_different_stages fatigue poor_appetite"),
    p("scarlet_fever", "Tinh hồng nhiệt", "Nhi", "child_patient sore_throat sandpaper_rash strawberry_tongue fever swollen_neck_nodes"),
    p("febrile_seizure", "Co giật do sốt", "Nhi", "child_patient fever seizure generalized_shaking brief_unconsciousness postictal_drowsiness first_seizure_red_flag"),
    p("pediatric_asthma", "Hen trẻ em", "Nhi", "child_patient recurrent_wheezing cough nighttime_cough exercise_trigger allergen_trigger breathing_difficulty"),
    p("atopic_dermatitis_child", "Viêm da cơ địa trẻ em", "Nhi", "child_patient itchy_rash dry_skin flexural_rash recurrent_rash family_atopy sleep_disturbance"),
    p("impetigo_child", "Chốc lở trẻ em", "Nhi", "child_patient honey_colored_crust skin_sores itchy_skin facial_rash spreading_lesions mild_fever"),
    p("mumps", "Quai bị", "Nhi", "child_patient parotid_swelling jaw_pain fever pain_when_chewing headache fatigue"),
    p("whooping_cough", "Ho gà", "Nhi", "child_patient coughing_fits inspiratory_whoop vomiting_after_cough apnea_cyanosis mild_fever"),
    p("pediatric_meningitis", "Viêm màng não trẻ em", "Nhi", "child_patient fever neck_stiffness severe_headache vomiting altered_consciousness non_blanching_rash bulging_fontanelle"),
    p("malnutrition_child", "Suy dinh dưỡng trẻ em", "Nhi", "child_patient poor_weight_gain muscle_wasting reduced_appetite recurrent_infection edema developmental_delay"),
    p("iron_deficiency_anemia_child", "Thiếu máu thiếu sắt trẻ em", "Nhi", "child_patient pallor fatigue poor_concentration pica poor_appetite rapid_heart_rate"),
    p("neonatal_jaundice", "Vàng da sơ sinh", "Nhi", "newborn_patient jaundice yellow_eyes poor_feeding lethargy dark_urine pale_stool high_pitched_cry"),

    # Răng Hàm Mặt
    p("pericoronitis", "Viêm lợi trùm răng khôn", "Răng Hàm Mặt", "wisdom_tooth_pain swollen_gum limited_mouth_opening bad_taste fever pain_when_swallowing"),
    p("temporomandibular_disorder", "Rối loạn khớp thái dương hàm", "Răng Hàm Mặt", "jaw_joint_pain jaw_clicking limited_mouth_opening pain_when_chewing facial_pain morning_jaw_stiffness"),
    p("recurrent_aphthous_ulcer", "Loét áp-tơ tái phát", "Răng Hàm Mặt", "painful_mouth_ulcer recurrent_oral_ulcer round_ulcer eating_pain no_fever healing_within_weeks"),
    p("oral_candidiasis", "Nấm miệng", "Răng Hàm Mặt", "white_oral_patches burning_mouth altered_taste removable_white_plaque immunosuppression_risk denture_risk"),
    p("oral_lichen_planus", "Lichen phẳng miệng", "Răng Hàm Mặt", "white_lacy_oral_patch burning_mouth oral_erosion bilateral_buccal_lesions chronic_course"),
    p("oral_cancer", "Ung thư khoang miệng", "Răng Hàm Mặt", "nonhealing_mouth_ulcer oral_lump unexplained_oral_bleeding difficulty_swallowing neck_lump tobacco_alcohol_risk weight_loss"),
    p("leukoplakia", "Bạch sản niêm mạc miệng", "Răng Hàm Mặt", "persistent_white_oral_patch non_scrapable_patch tobacco_risk painless_lesion rough_oral_patch"),
    p("dry_socket", "Viêm ổ răng khô", "Răng Hàm Mặt", "severe_pain_after_extraction empty_tooth_socket bad_breath bad_taste pain_to_ear exposed_bone"),
    p("impacted_wisdom_tooth", "Răng khôn mọc lệch", "Răng Hàm Mặt", "wisdom_tooth_pain jaw_swelling limited_mouth_opening food_trapping gum_inflammation recurrent_infection"),
    p("tooth_fracture", "Gãy răng", "Răng Hàm Mặt", "chipped_broken_tooth pain_when_biting tooth_sensitivity sharp_tooth_edge trauma_history visible_crack"),
    p("tooth_avulsion", "Răng bật khỏi ổ", "Răng Hàm Mặt", "missing_tooth_after_trauma bleeding_tooth_socket oral_trauma severe_tooth_pain lip_injury emergency_dental_sign"),
    p("malocclusion", "Sai khớp cắn", "Răng Hàm Mặt", "misaligned_teeth abnormal_bite chewing_difficulty speech_difficulty jaw_strain crowded_teeth"),
    p("bruxism", "Nghiến răng", "Răng Hàm Mặt", "tooth_wear morning_jaw_pain sleep_grinding tooth_sensitivity headache jaw_muscle_tenderness"),
    p("salivary_stone", "Sỏi tuyến nước bọt", "Răng Hàm Mặt", "meal_related_salivary_pain salivary_gland_swelling dry_mouth reduced_saliva gland_tenderness recurrent_swelling"),
    p("sialadenitis", "Viêm tuyến nước bọt", "Răng Hàm Mặt", "painful_salivary_swelling fever pus_from_salivary_duct dry_mouth meal_related_pain gland_redness"),

    # Tai Mũi Họng
    p("chronic_sinusitis", "Viêm xoang mạn", "Tai Mũi Họng", "chronic_nasal_congestion facial_pressure reduced_smell postnasal_drip thick_nasal_discharge symptoms_over_12_weeks"),
    p("nasal_polyps", "Polyp mũi", "Tai Mũi Họng", "persistent_nasal_blockage reduced_smell mouth_breathing postnasal_drip snoring bilateral_nasal_mass"),
    p("deviated_nasal_septum", "Vẹo vách ngăn mũi", "Tai Mũi Họng", "one_sided_nasal_obstruction recurrent_nosebleed facial_pain snoring mouth_breathing trauma_history"),
    p("epistaxis", "Chảy máu mũi", "Tai Mũi Họng", "nosebleed anterior_nasal_bleeding recurrent_bleeding nasal_dryness anticoagulant_risk high_blood_pressure"),
    p("peritonsillar_abscess", "Áp xe quanh amidan", "Tai Mũi Họng", "severe_one_sided_sore_throat muffled_voice limited_mouth_opening uvula_deviation fever drooling swallowing_difficulty"),
    p("laryngitis", "Viêm thanh quản", "Tai Mũi Họng", "hoarse_voice voice_loss sore_throat dry_cough throat_clearing recent_upper_respiratory_infection"),
    p("vocal_cord_nodule", "Hạt xơ dây thanh", "Tai Mũi Họng", "persistent_hoarseness voice_fatigue voice_overuse throat_discomfort reduced_vocal_range no_fever"),
    p("benign_positional_vertigo", "Chóng mặt tư thế lành tính", "Tai Mũi Họng", "brief_positional_vertigo spinning_sensation nausea triggered_by_head_movement nystagmus no_hearing_loss"),
    p("menieres_disease", "Bệnh Ménière", "Tai Mũi Họng", "episodic_vertigo fluctuating_hearing_loss tinnitus ear_fullness nausea recurrent_attacks"),
    p("labyrinthitis", "Viêm mê nhĩ", "Tai Mũi Họng", "acute_prolonged_vertigo hearing_loss nausea vomiting imbalance recent_viral_illness"),
    p("vestibular_neuritis", "Viêm thần kinh tiền đình", "Tai Mũi Họng", "acute_prolonged_vertigo imbalance nausea vomiting no_hearing_loss recent_viral_illness"),
    p("sudden_hearing_loss", "Điếc đột ngột", "Tai Mũi Họng", "sudden_one_sided_hearing_loss tinnitus ear_fullness dizziness onset_under_72_hours urgent_hearing_red_flag"),
    p("earwax_impaction", "Nút ráy tai", "Tai Mũi Họng", "blocked_ear reduced_hearing ear_fullness tinnitus ear_itching visible_earwax"),
    p("eustachian_tube_dysfunction", "Rối loạn vòi nhĩ", "Tai Mũi Họng", "ear_fullness popping_ear muffled_hearing pressure_change_trigger ear_discomfort recent_cold"),
    p("obstructive_sleep_apnea", "Ngưng thở khi ngủ do tắc nghẽn", "Tai Mũi Họng", "loud_snoring witnessed_apnea daytime_sleepiness morning_headache choking_during_sleep obesity_risk enlarged_tonsils"),
]


CONTEXT = {
    "demographic": "age_newborn age_infant age_child age_adolescent age_adult age_older_adult sex_female sex_male pregnancy_possible".split(),
    "duration": "duration_hours duration_1_3_days duration_4_7_days duration_1_4_weeks duration_over_4_weeks recurrent_course progressive_course intermittent_course".split(),
    "onset": "onset_sudden onset_gradual onset_after_meal onset_after_trauma onset_after_infection onset_after_exposure".split(),
    "severity": "severity_mild severity_moderate severity_severe limits_daily_activity wakes_from_sleep".split(),
    "exam": "exam_fever exam_tachycardia exam_low_oxygen exam_dehydration exam_pallor exam_jaundice exam_edema exam_tenderness exam_guarding exam_swelling exam_redness exam_discharge exam_lump exam_enlarged_nodes exam_abnormal_breath_sounds exam_neurologic_deficit exam_oral_lesion exam_hearing_reduced exam_nasal_obstruction exam_rash".split(),
    "risk": "risk_smoking risk_alcohol risk_diabetes risk_hypertension risk_obesity risk_immunosuppression risk_recent_surgery risk_recent_antibiotics risk_sick_contact risk_travel risk_poor_oral_hygiene risk_denture risk_voice_overuse risk_allergy risk_family_history risk_anticoagulant risk_nsaid_use risk_high_fat_diet risk_breastfeeding risk_unvaccinated".split(),
    "red_flag": "red_severe_breathing_difficulty red_low_oxygen red_unconsciousness red_new_seizure red_focal_weakness red_heavy_bleeding red_shock red_stiff_neck red_non_blanching_rash red_sudden_hearing_loss red_airway_obstruction red_inability_to_swallow red_persistent_vomiting red_severe_dehydration red_rapidly_growing_mass red_unexplained_weight_loss red_vomiting_blood red_black_stool red_board_like_abdomen red_neonate_fever".split(),
    "negative": "neg_no_fever neg_no_chest_pain neg_no_breathing_difficulty neg_no_vomiting neg_no_diarrhea neg_no_bleeding neg_no_weight_loss neg_no_neurologic_deficit neg_no_hearing_loss neg_no_ear_discharge neg_no_nasal_discharge neg_no_sore_throat neg_no_dental_trauma neg_no_facial_swelling neg_no_abdominal_guarding neg_no_dehydration neg_no_rash neg_no_pus_discharge neg_no_loss_of_consciousness neg_no_recent_infection".split(),
}


def normalize_base():
    return [tuple(x) for x in BASE_PROFILES] + ADDITIONAL


def specialty_context(specialty):
    return {
        "Nội tổng quát": ["exam_fever", "exam_tachycardia", "risk_smoking", "risk_diabetes", "risk_hypertension"],
        "Ngoại": ["exam_tenderness", "exam_guarding", "exam_swelling", "risk_recent_surgery", "red_shock"],
        "Nhi": ["age_child", "exam_fever", "risk_sick_contact", "risk_unvaccinated", "red_severe_dehydration"],
        "Răng Hàm Mặt": ["exam_oral_lesion", "exam_swelling", "risk_poor_oral_hygiene", "risk_diabetes", "red_airway_obstruction"],
        "Tai Mũi Họng": ["exam_hearing_reduced", "exam_nasal_obstruction", "exam_enlarged_nodes", "risk_allergy", "red_sudden_hearing_loss"],
    }[specialty]


def build():
    rng = random.Random(SEED)
    profiles = normalize_base()
    core_features = {s for _, _, _, core, _ in profiles for s in core}
    optional_counts = Counter(s for _, _, _, _, optional in profiles for s in optional)
    context = {x for values in CONTEXT.values() for x in values}
    # Preserve all core and context variables. Rank optional clinical features
    # by reuse across diseases, then by name, to keep the matrix at 450 columns.
    optional_budget = 450 - len(core_features | context)
    optional_ranked = sorted(
        (x for x in optional_counts if x not in core_features and x not in context),
        key=lambda x: (-optional_counts[x], x),
    )
    features = sorted(core_features | context | set(optional_ranked[:optional_budget]))
    rows = []
    for key, _, specialty, core, optional in profiles:
        seen = set()
        while len(seen) < SAMPLES_PER_DISEASE:
            active = {s for s in core if rng.random() < .93}
            active.update(s for s in optional if rng.random() < .50)
            while len(active & set(core)) < min(2, len(core)):
                active.add(rng.choice(core))
            # Demographics are explicit rather than inferred from disease labels.
            if specialty == "Nhi":
                active.add(rng.choices(["age_newborn", "age_infant", "age_child", "age_adolescent"], [2, 3, 6, 2])[0])
            else:
                active.add(rng.choices(["age_adolescent", "age_adult", "age_older_adult"], [1, 7, 3])[0])
            sex = rng.choice(["sex_female", "sex_male"])
            active.add(sex)
            if sex == "sex_female" and specialty != "Nhi" and rng.random() < .08:
                active.add("pregnancy_possible")
            active.add(rng.choice(CONTEXT["duration"]))
            active.add(rng.choice(CONTEXT["onset"]))
            active.add(rng.choice(CONTEXT["severity"]))
            for item in specialty_context(specialty):
                if rng.random() < .28:
                    active.add(item)
            if rng.random() < .12:
                active.add(rng.choice(CONTEXT["exam"]))
            if rng.random() < .10:
                active.add(rng.choice(CONTEXT["risk"]))
            if rng.random() < .035:
                active.add(rng.choice(CONTEXT["red_flag"]))
            # Explicit negative finding: distinct from an unrecorded/zero feature.
            if rng.random() < .45:
                active.add(rng.choice(CONTEXT["negative"]))
            signature = tuple(int(f in active) for f in features)
            if signature in seen:
                continue
            seen.add(signature)
            row = dict(zip(features, signature))
            row["prognosis"] = key
            rows.append(row)
    rng.shuffle(rows)
    return profiles, features, rows


def main():
    profiles, features, rows = build()
    with (ROOT / "training_data_5_specialties_v2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=features + ["prognosis"]); w.writeheader(); w.writerows(rows)
    with (ROOT / "disease_specialty_map_v2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["disease_key", "disease_name_vi", "specialty", "data_status"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for key, vi, specialty, _, _ in profiles:
            w.writerow({"disease_key": key, "disease_name_vi": vi, "specialty": specialty, "data_status": "synthetic_requires_clinical_review"})
    clinical = {s for _, _, _, core, optional in profiles for s in core + optional}
    with (ROOT / "feature_dictionary_v2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["feature", "feature_group", "value_definition"])
        for feature in features:
            group = "clinical_symptom_or_sign" if feature in clinical else next(k for k, v in CONTEXT.items() if feature in v)
            w.writerow([feature, group, "1=explicitly present; 0=not recorded (not necessarily absent)"])
    print(f"diseases={len(profiles)} specialties=5 features={len(features)} rows={len(rows)}")


if __name__ == "__main__":
    main()
