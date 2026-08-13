// Explanation sources shown to raters under blinded labels (Set A, Set B, ...).
// The label-to-method assignment is shuffled deterministically per rater+cohort,
// so the UI never reveals which system produced a set, but the mapping can be
// reconstructed for analysis (the true method key is stored in each submission).
// NOTE: ALEX ('with_shap_drlearner') is hidden in this round -- its explanations
// were already rated in the previous round. Re-add it here to restore it.
const EXPLANATION_METHODS = ['cot', 'hypogenic', 'researchagent'];

// Cohorts where only a subset of methods is available on the site.
const cohortMethodOverrides = {};

// FNV-1a hash for deterministic per-rater seeding
function hashString(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
}

// mulberry32 seeded PRNG
function mulberry32(seed) {
    return function () {
        seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function seededShuffle(array, rand) {
    const out = array.slice();
    for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
}

function assignBlindedSets(raterId, cohort) {
    const methods = cohortMethodOverrides[cohort] || EXPLANATION_METHODS;
    const rand = mulberry32(hashString(`${raterId}::${cohort}`));
    return seededShuffle(methods, rand).map((method, i) => ({
        label: `Set ${String.fromCharCode(65 + i)}`,
        method: method,
    }));
}

// Trial metadata
const trialInfo = {
    crash_2: {
        subgroup_analysis: "Prespecified subgroup analyses showed no strong evidence against homogeneity of treatment effect (unless p<0.001). No significant heterogeneity was observed for systolic blood pressure (p=0.51), Glasgow Coma Score at randomisation (p=0.50), type of injury (p=0.37), or time from injury to randomisation (p=0.11). Because digit preference reduced precision in the <1 h group, a post hoc early category of treatment at <=1 h from injury was also evaluated (Figure 3: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60835-5/fulltext#fig3).",
        abstract: "Background: Tranexamic acid can reduce bleeding in patients undergoing elective surgery. We assessed the effects of early administration of a short course of tranexamic acid on death, vascular occlusive events, and the receipt of blood transfusion in trauma patients.\n\nMethods: This randomised controlled trial was undertaken in 274 hospitals in 40 countries. 20 211 adult trauma patients with, or at risk of, significant bleeding were randomly assigned within 8 h of injury to either tranexamic acid (loading dose 1 g over 10 min then infusion of 1 g over 8 h) or matching placebo. Randomisation was balanced by centre, with an allocation sequence based on a block size of eight, generated with a computer random number generator. Both participants and study staff (site investigators and trial coordinating centre staff) were masked to treatment allocation. The primary outcome was death in hospital within 4 weeks of injury, and was described with the following categories: bleeding, vascular occlusion (myocardial infarction, stroke and pulmonary embolism), multiorgan failure, head injury, and other. All analyses were by intention to treat. \n\nFindings: 10 096 patients were allocated to tranexamic acid and 10 115 to placebo, of whom 10 060 and 10 067, respectively, were analysed. All-cause mortality was significantly reduced with tranexamic acid (1463 [14.5%] tranexamic acid group vs 1613 [16.0%] placebo group; relative risk 0.91, 95% CI 0.85-0.97; p=0.0035). The risk of death due to bleeding was significantly reduced (489 [4.9%] vs 574 [5.7%]; relative risk 0.85, 95% CI 0.76-0.96; p=0.0077).",
        interpretation: "Early tranexamic acid appears to reduce all-cause mortality and death due to bleeding in trauma patients when given soon after injury.",
        treatment: "Tranexamic acid (TXA)",
        outcome: "All-cause mortality at 28 days or in-hospital death",
        population: "Trauma patients with significant bleeding or at risk of significant hemorrhage",
        description: "CRASH-2 was a large international randomised placebo-controlled trial (N=20,211) evaluating the effect of early administration of tranexamic acid on death, vascular occlusive events, and blood transfusion in adult trauma patients with or at risk of significant bleeding, conducted across 274 hospitals in 40 countries.",
        link: "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60835-5/fulltext"
    },
    ist3: {
        subgroup_analysis: "Overall, little variation occurred in the adjusted effects of treatment in different subgroups. However, a significant difference did occur in the adjusted effect of treatment between patients older than 80 years and in patients 80 years or younger (p=0.027), suggesting greater benefit in those older than 80 years of age, contrary to expectations. Treatment appeared at least as effective in this age group as in younger patients. Significant trends towards larger effects of treatment in more severe strokes were also seen (as assessed by the NIHSS and by the predicted probability of a poor outcome). Benefit was greatest in patients treated within 3 h, but there was insufficient power to examine decay of benefit with time. Figure 3: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(12)60768-5/fulltext#fig3",
        background: "Thrombolysis is of net benefit in patients with acute ischaemic stroke, who are younger than 80 years of age and are treated within 4.5 h of onset. The third International Stroke Trial (IST-3) sought to determine whether a wider range of patients might benefit up to 6 h from stroke onset.",
        methods: "In this international, multicentre, randomised, open-treatment trial, patients were allocated to 0.9 mg/kg intravenous recombinant tissue plasminogen activator (rt-PA) or to control. The primary analysis was of the proportion of patients alive and independent, as defined by an Oxford Handicap Score (OHS) of 0-2 at 6 months. The study is registered, ISRCTN25765518.",
        findings: "3035 patients were enrolled by 156 hospitals in 12 countries. All of these patients were included in the analyses (1515 in the rt-PA group vs 1520 in the control group), of whom 1617 (53%) were older than 80 years of age. At 6 months, 554 (37%) patients in the rt-PA group versus 534 (35%) in the control group were alive and independent (OHS 0-2; adjusted odds ratio [OR] 1.13, 95% CI 0.95-1.35, p=0.181; a non-significant absolute increase of 14/1000, 95% CI -20 to 48). An ordinal analysis showed a significant shift in OHS scores; common OR 1.27 (95% CI 1.10-1.47, p=0.001). Fatal or non-fatal symptomatic intracranial haemorrhage within 7 days occurred in 104 (7%) patients in the rt-PA group versus 16 (1%) in the control group (adjusted OR 6.94, 95% CI 4.07-11.8; absolute excess 58/1000, 95% CI 44-72). More deaths occurred within 7 days in the rt-PA group (163 [11%]) than in the control group (107 [7%], adjusted OR 1.60, 95% CI 1.22-2.08, p=0.001; absolute increase 37/1000, 95% CI 17-57), but between 7 days and 6 months there were fewer deaths in the rt-PA group than in the control group, so that by 6 months, similar numbers, in total, had died (408 [27%] in the rt-PA group vs 407 [27%] in the control group).",
        interpretation: "For the types of patient recruited in IST-3, despite the early hazards, thrombolysis within 6 h improved functional outcome. Benefit did not seem to be diminished in elderly patients.",
        treatment: "IV alteplase (recombinant tissue plasminogen activator)",
        outcome: "Alive and independent (Oxford Handicap Score 0-2) at 6 months",
        population: "Acute ischemic stroke patients within 6 hours of symptom onset",
        description: "IST-3 was an international randomised open-label trial (N=3,035) testing whether IV alteplase (0.9 mg/kg) given within 6 hours of acute ischaemic stroke improved functional outcome at 6 months, enrolling patients across 156 hospitals in 12 countries, including those over 80 years of age.",
        link: "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(12)60768-5/fulltext"
    },
    sprint: {
        subgroup_analysis: "Prespecified subgroups of interest for all outcomes were defined according to status with respect to cardiovascular disease at baseline (yes vs. no), status with respect to chronic kidney disease at baseline (yes vs. no), sex, race (black vs. nonblack), age (<75 vs. >=75 years), and baseline systolic blood pressure in three levels (<=132 mm Hg, >132 to <145 mm Hg, and >=145 mm Hg). The effects of the intervention on the rate of the primary outcome and on the rate of death from any cause were consistent across the prespecified subgroups (Figure 4: https://www.nejm.org/doi/full/10.1056/NEJMoa1511939#f04, and Fig. S5 in the Supplementary Appendix: https://www.nejm.org/doi/full/10.1056/NEJMoa1511939#APPNEJMoa1511939SUP). There were no significant interactions between treatment and subgroup with respect to the primary outcome or death from any cause.",
        background: "The most appropriate targets for systolic blood pressure to reduce cardiovascular morbidity and mortality among persons without diabetes remain uncertain.",
        methods: "We randomly assigned 9361 persons with a systolic blood pressure of 130 mm Hg or higher and an increased cardiovascular risk, but without diabetes, to a systolic blood-pressure target of less than 120 mm Hg (intensive treatment) or a target of less than 140 mm Hg (standard treatment). The primary composite outcome was myocardial infarction, other acute coronary syndromes, stroke, heart failure, or death from cardiovascular causes.",
        findings: "At 1 year, the mean systolic blood pressure was 121.4 mm Hg in the intensive-treatment group and 136.2 mm Hg in the standard-treatment group. The intervention was stopped early after a median follow-up of 3.26 years owing to a significantly lower rate of the primary composite outcome in the intensive-treatment group than in the standard-treatment group (1.65% per year vs. 2.19% per year; hazard ratio with intensive treatment, 0.75; 95% confidence interval [CI], 0.64 to 0.89; P<0.001). All-cause mortality was also significantly lower in the intensive-treatment group (hazard ratio, 0.73; 95% CI, 0.60 to 0.90; P=0.003). Rates of serious adverse events of hypotension, syncope, electrolyte abnormalities, and acute kidney injury or failure, but not of injurious falls, were higher in the intensive-treatment group than in the standard-treatment group.",
        interpretation: "Among patients at high risk for cardiovascular events but without diabetes, targeting a systolic blood pressure of less than 120 mm Hg, as compared with less than 140 mm Hg, resulted in lower rates of fatal and nonfatal major cardiovascular events and death from any cause, although significantly higher rates of some adverse events were observed in the intensive-treatment group.",
        treatment: "Intensive blood pressure control (systolic BP target <120 mmHg)",
        outcome: "Composite of major cardiovascular events (MI, stroke, heart failure, cardiovascular death)",
        population: "Non-diabetic adults aged ≥50 with hypertension and increased cardiovascular risk",
        description: "SPRINT was a multicentre open-label randomised trial (N=9,361) comparing intensive systolic BP target (<120 mmHg) to standard target (<140 mmHg) in non-diabetic adults aged ≥50 with hypertension and at least one additional cardiovascular risk factor, conducted at 102 clinical sites in the United States.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa1511939"
    },
    accord: {
        subgroup_analysis: "There were no significant interactions among prespecified subgroups (see Section 17 in Supplementary Appendix 1): https://www.nejm.org/doi/suppl/10.1056/NEJMoa1001286/suppl_file/nejm_ac",
        background: "There is no evidence from randomized trials to support a strategy of lowering systolic blood pressure below 135 to 140 mm Hg in persons with type 2 diabetes mellitus. We investigated whether therapy targeting normal systolic pressure (i.e., <120 mm Hg) reduces major cardiovascular events in participants with type 2 diabetes at high risk for cardiovascular events.",
        methods: "A total of 4733 participants with type 2 diabetes were randomly assigned to intensive therapy, targeting a systolic pressure of less than 120 mm Hg, or standard therapy, targeting a systolic pressure of less than 140 mm Hg. The primary composite outcome was nonfatal myocardial infarction, nonfatal stroke, or death from cardiovascular causes. The mean follow-up was 4.7 years.",
        findings: "After 1 year, the mean systolic blood pressure was 119.3 mm Hg in the intensive-therapy group and 133.5 mm Hg in the standard-therapy group. The annual rate of the primary outcome was 1.87% in the intensive-therapy group and 2.09% in the standard-therapy group (hazard ratio with intensive therapy, 0.88; 95% confidence interval [CI], 0.73 to 1.06; P=0.20). The annual rates of death from any cause were 1.28% and 1.19% in the two groups, respectively (hazard ratio, 1.07; 95% CI, 0.85 to 1.35; P=0.55). The annual rates of stroke, a prespecified secondary outcome, were 0.32% and 0.53% in the two groups, respectively (hazard ratio, 0.59; 95% CI, 0.39 to 0.89; P=0.01). Serious adverse events attributed to antihypertensive treatment occurred in 77 of the 2362 participants in the intensive-therapy group (3.3%) and 30 of the 2371 participants in the standard-therapy group (1.3%) (P<0.001).",
        interpretation: "In patients with type 2 diabetes at high risk for cardiovascular events, targeting a systolic blood pressure of less than 120 mm Hg, as compared with less than 140 mm Hg, did not reduce the rate of a composite outcome of fatal and nonfatal major cardiovascular events.",
        treatment: "Intensive blood pressure control (systolic BP target <120 mmHg)",
        outcome: "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
        population: "Adults with type 2 diabetes and high cardiovascular risk",
        description: "ACCORD-BP was a randomised trial (N=4,733) embedded within the ACCORD study, comparing intensive systolic BP target (<120 mmHg) to standard target (<140 mmHg) in adults with type 2 diabetes and high cardiovascular risk, conducted at 77 clinical sites across the United States and Canada.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa1001286"
    },
    accord_glycemia: {
        subgroup_analysis: "For the primary outcome, there was some evidence of heterogeneity among prespecified subgroups, which suggested that patients in the intensive-therapy group who had not had a cardiovascular event before randomization (P for interaction=0.04) or whose baseline glycated hemoglobin level was 8.0% or less (P for interaction=0.03) may have had fewer fatal or nonfatal cardiovascular events than did patients in the standard-therapy group (Figure 3: https://www.nejm.org/doi/full/10.1056/NEJMoa0802743#f03). Preliminary nonprespecified exploratory analyses of episodes of severe hypoglycemia after randomization and differences in the use of drugs (including rosiglitazone), weight change, and other factors did not identify an explanation for the mortality finding.",
        background: "Epidemiologic studies have shown a relationship between glycated hemoglobin levels and cardiovascular events in patients with type 2 diabetes. We investigated whether intensive therapy to target normal glycated hemoglobin levels would reduce cardiovascular events in patients with type 2 diabetes who had either established cardiovascular disease or additional cardiovascular risk factors.",
        methods: "In this randomized study, 10,251 patients (mean age, 62.2 years) with a median glycated hemoglobin level of 8.1% were assigned to receive intensive therapy (targeting a glycated hemoglobin level below 6.0%) or standard therapy (targeting a level from 7.0 to 7.9%). Of these patients, 38% were women, and 35% had had a previous cardiovascular event. The primary outcome was a composite of nonfatal myocardial infarction, nonfatal stroke, or death from cardiovascular causes. The finding of higher mortality in the intensive-therapy group led to a discontinuation of intensive therapy after a mean of 3.5 years of follow-up.",
        findings: "At 1 year, stable median glycated hemoglobin levels of 6.4% and 7.5% were achieved in the intensive-therapy group and the standard-therapy group, respectively. During follow-up, the primary outcome occurred in 352 patients in the intensive-therapy group, as compared with 371 in the standard-therapy group (hazard ratio, 0.90; 95% confidence interval [CI], 0.78 to 1.04; P=0.16). At the same time, 257 patients in the intensive-therapy group died, as compared with 203 patients in the standard-therapy group (hazard ratio, 1.22; 95% CI, 1.01 to 1.46; P=0.04). Hypoglycemia requiring assistance and weight gain of more than 10 kg were more frequent in the intensive-therapy group (P<0.001).",
        interpretation: "As compared with standard therapy, the use of intensive therapy to target normal glycated hemoglobin levels for 3.5 years increased mortality and did not significantly reduce major cardiovascular events. These findings identify a previously unrecognized harm of intensive glucose lowering in high-risk patients with type 2 diabetes.",
        treatment: "Intensive glycemic control (HbA1c target <6.0%)",
        outcome: "First major cardiovascular event composite (nonfatal MI, nonfatal stroke, or cardiovascular death)",
        population: "Adults with type 2 diabetes at high cardiovascular risk",
        description: "ACCORD Glycemia was a randomized trial comparing intensive glucose lowering (HbA1c target <6.0%) versus standard control (target 7.0-7.9%) in adults with type 2 diabetes at high cardiovascular risk. The primary composite cardiovascular outcome was nonfatal MI, nonfatal stroke, or cardiovascular death; the intensive glycemia strategy was stopped early because of increased all-cause mortality.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa0802743"
    }
};

const specialtyToCohort = {
    emergency: 'crash_2',
    surgery: 'crash_2',
    neurology: 'ist3',
    'endocrinology and metabolism': 'accord_glycemia',
    cardiology: 'accord',
    'internal medicine': 'sprint',
};

function normalizeSpecialty(specialty) {
    return (specialty || '').trim().toLowerCase().replace(/\s+/g, ' ');
}

function getCohortForSpecialty(specialty) {
    return specialtyToCohort[normalizeSpecialty(specialty)] || '';
}

// Feature name mapping for clean display
const featureNameMap = {
    // IST-3 features
    'nihss': 'NIHSS Score',
    'age': 'Age',
    'weight': 'Weight',
    'glucose': 'Blood Glucose',
    'gcs_score_rand': 'GCS',
    'sbprand': 'Systolic Blood Pressure',
    'dbprand': 'Diastolic Blood Pressure',
    'gender': 'Sex',
    'antiplat_rand': 'Antiplatelet Usage',
    'atrialfib_rand': 'Atrial Fibrillation History',
    'infarct': 'Prior Infarct',
    'stroketype_1': 'Stroke Type: TACI',
    'stroketype_2': 'Stroke Type: PACI',
    'stroketype_3': 'Stroke Type: POCI',
    'stroketype_4': 'Stroke Type: LACI',
    'Stroke Type: TACI (Total Anterior Circulation Infarct)': 'Stroke Type: TACI',

    // CRASH-2 features
    'iage': 'Age',
    'isbp': 'Systolic Blood Pressure',
    'irr': 'Respiratory Rate',
    'icc': 'Injury Classification Code',
    'ihr': 'Heart Rate',
    'igcs': 'Glasgow Coma Scale',
    'ninjurytime': 'Time from Injury to Treatment',
    'isex': 'Sex',
    'iinjurytype': 'Injury Type',
    'iinjurytype_1': 'Injury Type: Blunt',
    'iinjurytype_2': 'Injury Type: Penetrating',
    'Injury Type: Penetrating': 'Injury Type: Penetrating',
    'Injury Type: Blunt': 'Injury Type: Blunt',

    // SPRINT features
    'sbp': 'Systolic Blood Pressure',
    'dbp': 'Diastolic Blood Pressure',
    'n_agents': 'Number of Antihypertensive Agents',
    'egfr': 'Estimated Glomerular Filtration Rate',
    'screat': 'Serum Creatinine',
    'chr': 'Total Cholesterol/HDL Ratio',
    'glur': 'Fasting Glucose',
    'hdl': 'HDL Cholesterol',
    'trr': 'Triglycerides',
    'umalcr': 'Urine Albumin-Creatinine Ratio',
    'bmi': 'Body Mass Index',
    'female': 'Sex',
    'race_black': 'Race (Black vs non-Black)',
    'smoke_3cat': 'Current Smoker',
    'aspirin': 'Aspirin Use',
    'statin': 'Statin Use',
    'sub_cvd': 'History of Cardiovascular Disease',
    'sub_ckd': 'Chronic Kidney Disease',

    // ACCORD features
    'baseline_age': 'Age',
    'hr': 'Heart Rate',
    'hba1c': 'HbA1c',
    'Hemoglobin A1c': 'Baseline HbA1c',
    'fpg': 'Fasting Plasma Glucose',
    'alt': 'Alanine Aminotransferase',
    'cpk': 'Creatine Phosphokinase',
    'potassium': 'Serum Potassium',
    'gfr': 'Glomerular Filtration Rate',
    'uacr': 'Urine Albumin-Creatinine Ratio',
    'chol': 'Total Cholesterol',
    'trig': 'Triglycerides',
    'vldl': 'VLDL Cholesterol',
    'ldl': 'LDL Cholesterol',
    'bp_med': 'Antihypertensive Medication Usage',
    'raceclass': 'Race',
    'cvd_hx_baseline': 'History of Cardiovascular Disease',
    'antiarrhythmic': 'Antiarrhythmic Medication Use',
    'anti_coag': 'Anticoagulant Usage',
    'x4smoke': 'Current Smoker',
};

function normalizeFeatureKey(name) {
    return (name || '').toLowerCase().replace(/[_\s]+/g, ' ').trim();
}

const normalizedFeatureNameMap = Object.fromEntries(
    Object.entries(featureNameMap).map(([key, value]) => [normalizeFeatureKey(key), value])
);

function getDisplayFeatureName(featureName) {
    // Check direct mapping first
    if (featureNameMap[featureName]) return featureNameMap[featureName];

    // Check normalized mapping to catch variants like "sub CVD" vs "sub_cvd"
    const normalized = normalizeFeatureKey(featureName);
    if (normalizedFeatureNameMap[normalized]) return normalizedFeatureNameMap[normalized];

    // Strip parenthesized raw variable names, e.g. "Injury classification code (icc)" -> "Injury classification code"
    const stripped = featureName.replace(/\s*\([^)]*\)\s*$/, '').trim();
    if (featureNameMap[stripped]) return featureNameMap[stripped];

    const normalizedStripped = normalizeFeatureKey(stripped);
    if (normalizedFeatureNameMap[normalizedStripped]) return normalizedFeatureNameMap[normalizedStripped];

    // Return the cleaned name (without raw variable in parentheses)
    return stripped;
}

// Rating criteria: 4 robustness gates + novelty bonus
// Aligned with judge_evaluation.py gate logic
const ratingGates = [
    {
        id: 'is_biologically_coherent',
        label: 'Q1: Logical Coherence',
        description: 'Does the explanation make sense and follow a clear line of reasoning? Look for a plausible biological, pharmacological, physiological, or clinical mechanism connecting the feature to different treatment effects. DISAGREE if the explanation is only a statistical/epidemiological claim, uses circular reasoning, or is logically inconsistent.'
    },
    {
        id: 'is_causally_plausible',
        label: 'Q2: Causal Plausibility',
        description: 'Is the proposed relationship between the patient characteristic and the treatment effect clinically or biologically plausible? AGREE example: "Patients with renal impairment clear the drug more slowly, leading to higher effective exposure and greater benefit." DISAGREE if the explanation collapses to absolute-risk amplification with constant relative risk reduction (e.g., "older patients benefit more" only because they have higher baseline event rates), or relies on post-treatment variables, reverse causality, or trivial severity proxies.'
    },
    {
        id: 'is_clinically_actionable',
        label: 'Q3: Clinical Actionability',
        description: 'Could the explanation help inform clinical interpretation or decision-making? Does it identify a patient subgroup with a distinct treatment recommendation that could be applied in practice?'
    },
    {
        id: 'is_literature_backed',
        label: 'Q4: Evidence Quality',
        description: 'Is the explanation well supported by the information provided and/or relevant clinical evidence? AGREE if you are aware of supporting evidence (RCT subgroup analyses, meta-analyses, clinical guidelines, or well-known clinical observations); DISAGREE if you have never encountered this explanation in the literature or clinical practice.'
    }
];

const noveltyBonus = {
    id: 'is_novel',
    label: 'Q5: Novelty',
    description: 'Does this explanation identify an underexplored mechanism or subgroup that is not already well-covered in existing clinical guidelines or major reviews?'
};

const API_BASE_URL = window.RATINGS_API_BASE_URL || 'http://localhost:8000';



let currentHypotheses = [];
let assignedSets = [];
let currentSetIndex = -1;
const submittedSetMethods = new Set();
// Per-set hypotheses and in-progress answers, keyed by set index, so raters
// can switch between sets without losing work; everything submits together.
const setStates = {};

function setLoadStatus(message, type = 'info') {
    const statusEl = document.getElementById('load-status');
    if (!statusEl) return;

    if (!message) {
        statusEl.style.display = 'none';
        statusEl.textContent = '';
        statusEl.style.color = '';
        return;
    }

    statusEl.style.display = 'block';
    statusEl.textContent = message;
    if (type === 'error') {
        statusEl.style.color = '#b00020';
    } else if (type === 'success') {
        statusEl.style.color = '#0a7a2f';
    } else {
        statusEl.style.color = '';
    }
}

// Load explanations when button is clicked
document.getElementById('load-btn').addEventListener('click', loadHypotheses);

async function loadHypotheses() {
    const expertise = document.getElementById('expertise-select').value;
    const specialty = document.getElementById('specialty-input').value;
    const raterId = document.getElementById('rater-id-input').value.trim();
    const raterIdPattern = /^[a-zA-Z0-9_-]{3,64}$/;

    if (!specialty) {
        setLoadStatus('Please select your specialty.', 'error');
        alert('Please select your specialty');
        return;
    }

    const cohort = getCohortForSpecialty(specialty);

    if (!cohort) {
        setLoadStatus('Please select a specialty mapped to a trial cohort.', 'error');
        alert('Please select a specialty with a mapped trial cohort');
        return;
    }

    if (!expertise) {
        setLoadStatus('Please select your clinical expertise level.', 'error');
        alert('Please select your clinical expertise level');
        return;
    }

    if (!raterId || !raterIdPattern.test(raterId)) {
        setLoadStatus('Please enter a valid anonymous ID (3-64 chars; letters, numbers, _ or -). Do not use a recognizable personal ID.', 'error');
        alert('Please enter a valid anonymous ID (3-64 chars; letters, numbers, _ or -). Do not use a recognizable personal ID.');
        return;
    }

    assignedSets = assignBlindedSets(raterId, cohort);
    currentSetIndex = -1;
    submittedSetMethods.clear();
    Object.keys(setStates).forEach(k => delete setStates[k]);
    renderSetBar();
    await selectSet(0);
}

function allGatesList() {
    return [...ratingGates, noveltyBonus];
}

// Snapshot the current set's answers from the DOM into setStates
function saveCurrentSetAnswers() {
    if (currentSetIndex < 0 || !currentHypotheses.length) return;
    const state = setStates[currentSetIndex];
    if (!state) return;
    state.answers = currentHypotheses.map((hyp, index) => {
        const entry = {};
        allGatesList().forEach(gate => {
            const trueBtn = document.getElementById(`${gate.id}-${index}-true`);
            const falseBtn = document.getElementById(`${gate.id}-${index}-false`);
            entry[gate.id] = trueBtn && trueBtn.classList.contains('active') ? true
                : (falseBtn && falseBtn.classList.contains('active') ? false : null);
            const commentEl = document.getElementById(`${gate.id}-comments-${index}`);
            entry[`${gate.id}_comments`] = commentEl ? commentEl.value : '';
        });
        const commentsEl = document.getElementById(`comments-${index}`);
        entry.comments = commentsEl ? commentsEl.value : '';
        return entry;
    });
}

// Re-apply saved answers to the freshly rendered cards of a set
function restoreSetAnswers(index) {
    const state = setStates[index];
    if (!state || !state.answers) return;
    state.answers.forEach((entry, hypIndex) => {
        allGatesList().forEach(gate => {
            if (entry[gate.id] === true || entry[gate.id] === false) {
                setGate(gate.id, hypIndex, entry[gate.id]);
            }
            const commentEl = document.getElementById(`${gate.id}-comments-${hypIndex}`);
            if (commentEl && entry[`${gate.id}_comments`]) {
                commentEl.value = entry[`${gate.id}_comments`];
            }
        });
        const commentsEl = document.getElementById(`comments-${hypIndex}`);
        if (commentsEl && entry.comments) {
            commentsEl.value = entry.comments;
        }
    });
}

function isSetComplete(index) {
    const state = setStates[index];
    if (!state || !state.answers || !state.hypotheses || !state.hypotheses.length) return false;
    return state.answers.every(entry =>
        allGatesList().every(gate => entry[gate.id] === true || entry[gate.id] === false)
    );
}

async function selectSet(index) {
    if (index === currentSetIndex) return;
    saveCurrentSetAnswers();
    currentSetIndex = index;
    updateSetBar();
    await loadSet(index);
}

async function loadSet(index) {
    const set = assignedSets[index];
    const expertise = document.getElementById('expertise-select').value;
    const specialty = document.getElementById('specialty-input').value;
    const raterId = document.getElementById('rater-id-input').value.trim();
    const cohort = getCohortForSpecialty(specialty);

    const filePath = `agent/${cohort}/gpt-5-mini/${set.method}/seed_0/hypotheses.json`;
    setLoadStatus('Loading explanations...');

    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Normalize different JSON formats into unified hypothesis list;
        // seed sampling by rater+cohort+method so refreshes show the same items
        const rand = mulberry32(hashString(`${raterId}::${cohort}::${set.method}`));
        const hypotheses = normalizeHypotheses(data, set.method, rand);

        setStates[index] = setStates[index] || {};
        setStates[index].hypotheses = hypotheses;

        displayTrialInfo(cohort);
        displayHypotheses(hypotheses, cohort, set.method, expertise, specialty, raterId, set.label);
        restoreSetAnswers(index);
        updateSetBar();
        setLoadStatus(`Loaded ${hypotheses.length} explanations for ${set.label}.`, 'success');

    } catch (error) {
        console.error(`Error loading explanations from ${filePath}:`, error);
        setLoadStatus(`Error loading explanations for ${set.label}: ${error.message}`, 'error');
        const explanationsIntro = document.getElementById('explanations-intro');
        if (explanationsIntro) {
            explanationsIntro.style.display = 'none';
        }
        const container = document.getElementById('hypotheses-container');
        container.innerHTML = `
            <div class="error">
                <strong>Error loading explanations for ${set.label}:</strong> ${error.message}
            </div>
        `;
    }
}

// Normalize different method JSON formats into a common structure
function normalizeHypotheses(data, method, rand = Math.random) {
    if (method === 'hypogenic') {
        // HypoGeniC uses a dict keyed by hypothesis text
        return Object.values(data.hypotheses || {}).map((h, i) => ({
            feature_name: h.subgroup_rule?.feature || `Hypothesis ${i + 1}`,
            hypothesis_text: h.hypothesis,
            mechanisms: [{ description: h.hypothesis }],
            subgroup_rule: h.subgroup_rule,
            recommendation: h.recommendation,
            importance_rank: i + 1
        }));
    }
    // ALEX, CoT, ResearchAgent all use feature_hypotheses array
    let hypotheses = (data.feature_hypotheses || []).map((h, i) => ({
        feature_name: h.feature_name,
        mechanisms: (h.mechanisms || []).map(m => ({ description: m.description })),
        importance_rank: h.importance_rank || i + 1
    }));

    // For ResearchAgent, split each mechanism into its own card, deduplicate, then sample 5
    if (method === 'researchagent') {
        const seen = new Set();
        const split = [];
        for (const h of hypotheses) {
            for (const m of h.mechanisms) {
                const desc = m.description.trim();
                if (!seen.has(desc)) {
                    seen.add(desc);
                    split.push({
                        feature_name: h.feature_name,
                        mechanisms: [{ description: desc }],
                        importance_rank: 0
                    });
                }
            }
        }
        // Shuffle and pick 5
        for (let i = split.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [split[i], split[j]] = [split[j], split[i]];
        }
        hypotheses = split.slice(0, 5);
        hypotheses.forEach((h, i) => h.importance_rank = i + 1);
    }

    return hypotheses;
}

function renderSetBar() {
    const bar = document.getElementById('set-selector');
    const buttonsEl = document.getElementById('set-buttons');
    if (!bar || !buttonsEl) return;

    buttonsEl.innerHTML = '';
    assignedSets.forEach((set, i) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'set-btn';
        btn.id = `set-btn-${i}`;
        btn.textContent = set.label;
        btn.addEventListener('click', () => selectSet(i));
        buttonsEl.appendChild(btn);
    });

    bar.style.display = assignedSets.length > 1 ? 'block' : 'none';
    updateSetBar();
}

function updateSetBar() {
    assignedSets.forEach((set, i) => {
        const btn = document.getElementById(`set-btn-${i}`);
        if (!btn) return;
        const done = submittedSetMethods.has(set.method) || isSetComplete(i);
        btn.classList.toggle('active', i === currentSetIndex);
        btn.classList.toggle('submitted', done);
        btn.textContent = done ? `${set.label} ✓` : set.label;
    });
}

function displayTrialInfo(cohort) {
    const info = trialInfo[cohort];

    const trialDisplayNames = {
        crash_2: 'CRASH-2',
        ist3: 'IST-3',
        sprint: 'SPRINT',
        accord: 'ACCORD-BP',
        accord_glycemia: 'ACCORD Glycemia',
    };

    const extractSection = (abstractText, sectionName) => {
        if (!abstractText) return '';
        const regex = new RegExp(`${sectionName}:\\s*([\\s\\S]*?)(?=\\n\\n(?:Background|Methods|Findings|Interpretation):|$)`, 'i');
        const match = abstractText.match(regex);
        return match ? match[1].trim() : '';
    };

    const background = info.background
        || extractSection(info.abstract, 'Background')
        || info.description
        || '';
    const methods = info.methods
        || extractSection(info.abstract, 'Methods')
        || `Trial population: ${info.population}.`;
    const findings = info.findings
        || extractSection(info.abstract, 'Findings')
        || `Primary outcome: ${info.outcome}.`;
    const interpretation = info.interpretation
        || extractSection(info.abstract, 'Interpretation')
        || `Clinical interpretation: ${info.treatment} evaluated in ${cohort.replace('_', ' ').toUpperCase()} for outcome improvement.`;
    const subgroupAnalysis = info.subgroup_analysis || 'Subgroup analysis text will be added.';

    const toHtmlWithLinks = (text) => {
        const escapeHtml = (s) => s
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        const parts = (text || '').split(/(https?:\/\/\S+)/g);
        const output = [];

        for (let idx = 0; idx < parts.length; idx++) {
            const part = parts[idx];

            // Even indices are normal text; odd indices are URL candidates.
            if (idx % 2 === 0) {
                output.push(escapeHtml(part));
                continue;
            }

            let url = part;
            let trailing = '';

            // Remove only trailing punctuation not part of the URL.
            while (url.length > 0 && /[),.;!?]$/.test(url)) {
                const lastChar = url.slice(-1);

                if (lastChar === ')') {
                    const openCount = (url.match(/\(/g) || []).length;
                    const closeCount = (url.match(/\)/g) || []).length;
                    if (closeCount <= openCount) {
                        break;
                    }
                }

                trailing = lastChar + trailing;
                url = url.slice(0, -1);
            }

            let linkText = 'Open link';
            const prevRaw = parts[idx - 1] || '';
            const figureMatch = prevRaw.match(/(Figure\s*\d+|Fig\.\s*S?\d+)\s*:\s*$/i);
            const appendixMatch = prevRaw.match(/(Supplementary Appendix)\s*:\s*$/i);

            if (figureMatch) {
                // Remove duplicated figure label from plain text and make it the link text.
                const prevWithoutLabel = prevRaw.slice(0, figureMatch.index);
                output[output.length - 1] = escapeHtml(prevWithoutLabel);
                linkText = figureMatch[1];
            } else if (appendixMatch) {
                const prevWithoutLabel = prevRaw.slice(0, appendixMatch.index);
                output[output.length - 1] = escapeHtml(prevWithoutLabel);
                linkText = appendixMatch[1];
            }

            output.push(`<a href="${url}" target="_blank" rel="noopener noreferrer">${escapeHtml(linkText)}</a>${escapeHtml(trailing)}`);
        }

        return output.join('');
    };

    document.getElementById('trial-name').textContent = trialDisplayNames[cohort] || cohort;
    const abstractParts = [background, methods, findings, interpretation].filter(Boolean);
    document.getElementById('trial-abstract').textContent = abstractParts.join(' ');
    document.getElementById('trial-subgroup').innerHTML = toHtmlWithLinks(subgroupAnalysis);

    const linkEl = document.getElementById('trial-link');
    linkEl.href = info.link;
    linkEl.textContent = 'View Publication';
    document.getElementById('trial-info').style.display = 'block';
}

function displayHypotheses(hypotheses, cohort, method, expertise, specialty, raterId, setLabel) {
    currentHypotheses = hypotheses;

    const container = document.getElementById('hypotheses-container');
    container.innerHTML = '';

    hypotheses.forEach((hyp, index) => {
        const card = createHypothesisCard(hyp, index);
        container.appendChild(card);
    });

    const explanationsIntro = document.getElementById('explanations-intro');
    if (explanationsIntro) {
        explanationsIntro.style.display = 'block';
    }

    const explanationsHeading = document.getElementById('explanations-heading');
    if (explanationsHeading) {
        const setPrefix = (setLabel && assignedSets.length > 1) ? `${setLabel} — ` : '';
        explanationsHeading.textContent = `${setPrefix}Explanations to Evaluate (${hypotheses.length})`;
    }

    document.getElementById('summary-section').style.display = 'block';
}

function createHypothesisCard(hypothesis, index) {
    const card = document.createElement('div');
    card.className = 'hypothesis-card';
    card.id = `hyp-${index}`;

    card.innerHTML = `
        <div class="hypothesis-header">
            <div class="hypothesis-title">${getDisplayFeatureName(hypothesis.feature_name)}</div>
        </div>

        <div class="hypothesis-content">
            <div class="content-section">
                ${hypothesis.mechanisms.map(m => `
                    <div class="mechanism-item">
                        ${m.description}
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="rating-section">
            <p class="gate-instructions">For each criterion, select AGREE or DISAGREE.</p>
            ${createGateInputs(index)}

            <div class="rating-group">
                <label class="rating-label">(Optional) Overall comments on this explanation</label>
                <textarea id="comments-${index}" placeholder="Any additional thoughts not captured by the per-criterion comments above..."></textarea>
            </div>
        </div>
    `;

    return card;
}

function createGateInputs(hypIndex) {
    const gateHTML = ratingGates.map(gate => `
        <div class="rating-group gate-group">
            <label class="rating-label">${gate.label}</label>
            <div class="rating-description">${gate.description}</div>
            <div class="gate-toggle">
                <button type="button" class="gate-btn gate-btn-true" id="${gate.id}-${hypIndex}-true"
                    onclick="setGate('${gate.id}', ${hypIndex}, true)">
                    AGREE
                </button>
                <button type="button" class="gate-btn gate-btn-false" id="${gate.id}-${hypIndex}-false"
                    onclick="setGate('${gate.id}', ${hypIndex}, false)">
                    DISAGREE
                </button>
            </div>
            <div class="rating-group gate-comment-group">
                <label class="rating-label">(Optional) Additional comments for ${(gate.label.match(/^Q\d+/i) || ['this question'])[0]} &mdash; e.g., explanation for your rating</label>
                <textarea id="${gate.id}-comments-${hypIndex}"></textarea>
            </div>
        </div>
    `).join('');

    const noveltyHTML = `
        <div class="rating-group gate-group novelty-group">
            <label class="rating-label">${noveltyBonus.label}</label>
            <div class="rating-description">${noveltyBonus.description}</div>
            <div class="gate-toggle">
                <button type="button" class="gate-btn gate-btn-true" id="${noveltyBonus.id}-${hypIndex}-true"
                    onclick="setGate('${noveltyBonus.id}', ${hypIndex}, true)">
                    AGREE
                </button>
                <button type="button" class="gate-btn gate-btn-false" id="${noveltyBonus.id}-${hypIndex}-false"
                    onclick="setGate('${noveltyBonus.id}', ${hypIndex}, false)">
                    DISAGREE
                </button>
            </div>
            <div class="rating-group gate-comment-group">
                <label class="rating-label">(Optional) Additional comments for Q5 &mdash; e.g., explanation for your rating</label>
                <textarea id="${noveltyBonus.id}-comments-${hypIndex}"></textarea>
            </div>
        </div>
    `;

    return gateHTML + noveltyHTML;
}

function setGate(gateId, hypIndex, value) {
    const trueBtn = document.getElementById(`${gateId}-${hypIndex}-true`);
    const falseBtn = document.getElementById(`${gateId}-${hypIndex}-false`);

    trueBtn.classList.remove('active');
    falseBtn.classList.remove('active');
    (value ? trueBtn : falseBtn).classList.add('active');

    trueBtn.dataset.value = value ? 'true' : '';
    falseBtn.dataset.value = value ? '' : 'false';

    // Keep the saved snapshot and the set-bar checkmarks in sync as the
    // rater answers questions
    if (currentSetIndex >= 0 && setStates[currentSetIndex]) {
        saveCurrentSetAnswers();
        updateSetBar();
    }
}

// Submit ratings
document.getElementById('submit-btn').addEventListener('click', submitRatings);

// Build one submission payload per set (same record shape as previous rounds:
// one record per method, plus set_label). Returns {missing: ...} if a question
// is unanswered.
function buildSetPayload(index, common) {
    const state = setStates[index];
    const set = assignedSets[index];
    if (!state || !state.hypotheses || !state.answers) {
        return { missing: { setIndex: index, hypIndex: 0, gate: ratingGates[0], featureName: '' } };
    }

    const ratingsList = [];
    for (let h = 0; h < state.hypotheses.length; h++) {
        const hyp = state.hypotheses[h];
        const entry = state.answers[h] || {};
        const featureRating = {
            feature_name: hyp.feature_name,
            feature_index: h,
        };

        for (const gate of allGatesList()) {
            const value = entry[gate.id];
            if (value !== true && value !== false) {
                return {
                    missing: {
                        setIndex: index,
                        hypIndex: h,
                        gate,
                        featureName: getDisplayFeatureName(hyp.feature_name),
                    }
                };
            }
            featureRating[gate.id] = value;
            const gateComment = (entry[`${gate.id}_comments`] || '').trim();
            if (gateComment) {
                featureRating[`${gate.id}_comments`] = gateComment;
            }
        }

        const comments = (entry.comments || '').trim();
        if (comments) {
            featureRating.comments = comments;
        }

        ratingsList.push(featureRating);
    }

    return {
        payload: {
            ...common,
            method: set.method,
            set_label: set.label,
            timestamp: new Date().toISOString(),
            ratings: ratingsList,
        }
    };
}

async function submitRatings() {
    saveCurrentSetAnswers();

    const expertise = document.getElementById('expertise-select').value;
    const specialty = document.getElementById('specialty-input').value;
    const raterId = document.getElementById('rater-id-input').value.trim();
    const raterIdPattern = /^[a-zA-Z0-9_-]{3,64}$/;

    if (!expertise) {
        alert('Please select your clinical expertise level');
        return;
    }

    if (!specialty) {
        alert('Please select your specialty');
        return;
    }

    const cohort = getCohortForSpecialty(specialty);

    if (!cohort) {
        alert('Selected specialty is not mapped to a trial cohort');
        return;
    }

    if (!raterId || !raterIdPattern.test(raterId)) {
        alert('Please enter a valid anonymous ID (3-64 chars; letters, numbers, _ or -). Do not use a recognizable personal ID.');
        return;
    }

    const common = {
        expertise: expertise,
        specialty: specialty,
        rater_id: raterId,
        cohort: cohort,
    };

    // All sets must be fully rated before anything is submitted
    const payloads = [];
    for (let i = 0; i < assignedSets.length; i++) {
        const result = buildSetPayload(i, common);
        if (result.missing) {
            const set = assignedSets[result.missing.setIndex];
            const where = result.missing.featureName
                ? `First missing: ${result.missing.gate.label} for ${result.missing.featureName}.`
                : 'That set has not been rated yet.';
            alert(
                `Please complete all ratings for ${set.label} before submitting. ${where}`
            );
            await selectSet(result.missing.setIndex);
            const targetEl = document.getElementById(
                `${result.missing.gate.id}-${result.missing.hypIndex}-true`
            );
            if (targetEl) {
                targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }
        payloads.push(result.payload);
    }

    const submitBtn = document.getElementById('submit-btn');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';

    try {
        const submissionIds = [];
        for (let i = 0; i < payloads.length; i++) {
            const set = assignedSets[i];
            // Skip sets already stored (retry after a partial failure)
            if (submittedSetMethods.has(set.method)) continue;

            const response = await fetch(`${API_BASE_URL}/api/ratings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payloads[i])
            });

            if (!response.ok) {
                const body = await response.text();
                throw new Error(`Submit failed for ${set.label} (${response.status}): ${body}`);
            }

            const result = await response.json();
            submittedSetMethods.add(set.method);
            submissionIds.push(result.submission_id);
        }

        alert(
            `Thank you for completing the survey.\n\n` +
            `Your ratings for all ${assignedSets.length} explanation set(s) were submitted successfully.\n\n` +
            `You can now close this website.`
        );
    } catch (error) {
        alert(
            `Could not submit all ratings to the backend. Any sets already submitted ` +
            `have been saved; click Submit again to retry the remaining ones.\n\n` +
            `Error: ${error.message}`
        );
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
}
