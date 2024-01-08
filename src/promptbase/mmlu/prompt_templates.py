from liquid import Template

logprobs_0shots = {
    "prompt_name": "logprobs_0shots",
    "response_type": "logprobs",
    "prompt": Template(
        """## Question
{{ question }}
## Task
Answer the above question with only 1 letter (such as A)

## Answer
"""
    ),
}

logprobs_5shots = {
    "prompt_name": "logprobs_5shots",
    "response_type": "logprobs",
    "prompt": Template(
        """{% for item in examples %}## Question
{{ item.question }}
## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
    ),
    "examples": [
        {
            "question": """A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow
""",
            "answer": "A",
        },
        {
            "question": """Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid
""",
            "answer": "C",
        },
        {
            "question": """A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction
""",
            "answer": "D",
        },
        {
            "question": """You are counseling a pregnant woman who plans to breast-feed exclusively regarding her newborn's nutritional requirements. The child was born at home and the mother only plans for her newborn to receive vaccinations but no other routine medical care. Which vitamins should be given to the newborn?

A. Folic acid
B. Vitamin K
C. Vitamin D
D. Vitamin K and Vitamin D
""",
            "answer": "D",
        },
        {
            "question": """An investigator is studying nutritional deficiencies in humans. A group of healthy volunteers are started on a diet deficient in pantothenic acid. After 4 weeks, several of the volunteers develop irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms are fully reversed after reintroduction of pantothenic acid to their diet. The function of which of the following enzymes was most likely impaired in the volunteers during the study?

A. Methionine synthase
B. Alpha-ketoglutarate dehydrogenase
C. Glutathione reductase
D. Dopamine beta-hydroxylase
""",
            "answer": "B",
        },
    ],
}

few_shots_prompt = Template(
    """{% for item in examples %}## Question
{{ item.question }}
## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
)

few_shots_prompt_with_context = Template(
    """{% for item in examples %}## Question
{{ item.question }}
## Context
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Context
"""
)

google_chain_of_thoughts = {
    "prompt_name": "google_chain_of_thoughts",
    "response_type": "MC",
    "prompt": few_shots_prompt,
    "examples": [
        {
            "question": """A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow
""",
            "answer": """Explanation: Let's solve this step-by-step, referring to authoritative sources as needed. A supracondylar fracture of the humerus with anterior displacement can potentially injure the brachial artery which runs anterior to the humerus and is responsible for the main blood supply to the forearm, wrist, and hand. This injury can cause an absent distal radial pulse due to decreased blood flow.
Answer: [A].""",
        },
        {
            "question": """Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid
""",
            "answer": """Explanation: Let's solve this step-by-step, referring to authoritative sources as needed. This patient's symptoms of bilateral non-pitting edema following a surgical procedure involving removal of regional lymph nodes suggest the diagnosis of lymphedema. Lymphedema is a condition characterized by localized fluid retention and tissue swelling caused by a compromised lymphatic system. The fluid accumulated in lymphedema is not merely protein-rich, but also lipid-rich. This occurs due to the blockage of lymphatic vessels which normally serve to drain both lipids and proteins. The interstitial fluid in lymphedema contains a high concentration of proteins, lipids, cells, and cellular debris.
Answer: [C].""",
        },
        {
            "question": """A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction
""",
            "answer": """Explanation: This patient's findings of pallor, tachycardia, microcytic hypochromic anemia with anisocytosis, and a positive direct antiglobulin test (Coombs test) in the setting of an antihypertensive medication use are suggestive of drug-induced immune hemolytic anemia. This condition occurs when drugs (eg, ceftriaxone, penicillin, quinidine, methyldopa) bind to the red blood cell membrane, causing production of anti-drug antibodies that result in hemolysis. The direct antiglobulin test is positive due to the presence of IgG on the red blood cell membrane. Given the recent medication initiation, cessation of the medication is necessary to stop the hemolysis.
Answer: [D].""",
        },
        {
            "question": """You are counseling a pregnant woman who plans to breast-feed exclusively regarding her newborn's nutritional requirements. The child was born at home and the mother only plans for her newborn to receive vaccinations but no other routine medical care. Which vitamins should be given to the newborn?

A. Folic acid
B. Vitamin K
C. Vitamin D
D. Vitamin K and Vitamin D
""",
            "answer": """Explanation: Exclusively breastfed infants are at risk of vitamin D deficiency because breast milk is low in vitamin D, and homebound infants may not get adequate sunlight exposure. In addition, vitamin K deficiency can occur in newborns due to inadequate transplacental transfer, low levels in breast milk, and sterile newborn gut (vitamin K is synthesized by gut bacteria). Thus, infants should receive prophylactic Vitamin K at birth to prevent Vitamin K deficiency bleeding, and Vitamin D supplementation should be started within the first few days of life and continue throughout infancy for those who are exclusively or partially breastfed.

Answer: [D].""",
        },
        {
            "question": """An investigator is studying nutritional deficiencies in humans. A group of healthy volunteers are started on a diet deficient in pantothenic acid. After 4 weeks, several of the volunteers develop irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms are fully reversed after reintroduction of pantothenic acid to their diet. The function of which of the following enzymes was most likely impaired in the volunteers during the study?

A. Methionine synthase
B. Alpha-ketoglutarate dehydrogenase
C. Glutathione reductase
D. Dopamine beta-hydroxylase
""",
            "answer": """Explanation: Pantothenic acid, also known as vitamin B5, is a component of coenzyme A (CoA), which is necessary for the metabolism of carbohydrates, amino acids, and lipids. CoA is also involved in the citric acid cycle (or Krebs cycle), where it forms a complex with acetyl groups (acetyl-CoA) to be oxidized for energy production. Alpha-ketoglutarate dehydrogenase is a key enzyme in the citric acid cycle and requires CoA for its function. Thus, a deficiency in pantothenic acid impairs the function of alpha-ketoglutarate dehydrogenase and disrupts the citric acid cycle, leading to the symptoms observed in the volunteers.

Answer: [B].""",
        },
    ],
}

# The example is generated by
# Let's think step by step. Then, write your answer by [A], [B], [C] or [D].
gpt_chain_of_thoughts = {
    "prompt_name": "gpt_chain_of_thoughts",
    "response_type": "MC",
    "prompt": few_shots_prompt,
    "examples": [
        {
            "question": """A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow
""",
            "answer": """A supracondylar fracture of the humerus, especially with anterior displacement of the proximal fragment, is a common and serious injury in children. It can cause injury to the brachial artery, leading to compromised blood flow to the forearm and hand. Therefore, the most likely additional finding in this case would be an absent distal radial pulse due to this potential vascular compromise. This is a serious complication and requires immediate orthopedic consultation and likely surgical intervention to prevent long-term damage.

So, the answer is [A].""",
        },
        {
            "question": """Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid
""",
            "answer": """This patient's bilateral, symmetric lower extremity swelling extending to the thighs, nonpitting quality, and absence of systemic symptoms (eg, shortness of breath, orthopnea) are suggestive of lymphedema. Lymphedema develops due to a blockage in the lymphatic system that prevents lymph fluid from draining adequately. In this case, the patient's recent surgery (low anterior resection for rectal cancer) probably involved extensive lymph node dissection, leading to secondary lymphedema.

Microscopically, lymph is a protein-rich fluid that also contains lipids, lymphocytes, and other substances. It does not typically contain significant amounts of glycosaminoglycans or hemosiderin.

Therefore, the answer is [C].""",
        },
        {
            "question": """A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction
""",
            "answer": """The patient's microcytic, hypochromic anemia and a high reticulocyte count are indicative of increased erythropoiesis, which, in this clinical scenario, is likely due to hemolysis. The positive direct antiglobulin test (Coomb's test) confirms that the hemolysis is immune-mediated. Given that other causes of immune hemolysis such as autoimmune disorders and infections are unlikely in this patient, the most probable cause of hemolysis is a drug-induced immune hemolytic reaction. Drugs can cause immune hemolysis by acting as haptens that bind to the red blood cell membrane, causing the formation of antibodies against the drug-cell complex (immune complex type) or by inducing the formation of autoantibodies against red blood cells (autoimmune type). While any drug can potentially cause immune hemolysis, common culprits include penicillins, cephalosporins, quinidine, and antihypertensives like methyldopa. 

Hence, the answer is [D].""",
        },
        {
            "question": """You are counseling a pregnant woman who plans to breast-feed exclusively regarding her newborn's nutritional requirements. The child was born at home and the mother only plans for her newborn to receive vaccinations but no other routine medical care. Which vitamins should be given to the newborn?

A. Folic acid
B. Vitamin K
C. Vitamin D
D. Vitamin K and Vitamin D
""",
            "answer": """The American Academy of Pediatrics (AAP) recommends that all newborns receive a single dose of intramuscular vitamin K at birth. This is because newborns have low stores of vitamin K at birth, leading to a risk of vitamin K deficiency bleeding, also known as hemorrhagic disease of the newborn. This is a potentially life-threatening condition that can cause bleeding into the brain. Because this baby was born at home and the mother plans to avoid routine medical care, this baby may not have received this essential vitamin.

Additionally, the AAP recommends that all infants and children, including those who are breastfed, have a minimum intake of 400 IU/day of vitamin D beginning soon after birth. While breast milk is the best source of most nutrients, it is usually deficient in vitamin D.

The other option, folic acid, is not routinely supplemented in newborns as they usually receive adequate amounts from breast milk or formula.

Therefore, the answer is [D].""",
        },
        {
            "question": """An investigator is studying nutritional deficiencies in humans. A group of healthy volunteers are started on a diet deficient in pantothenic acid. After 4 weeks, several of the volunteers develop irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms are fully reversed after reintroduction of pantothenic acid to their diet. The function of which of the following enzymes was most likely impaired in the volunteers during the study?

A. Methionine synthase
B. Alpha-ketoglutarate dehydrogenase
C. Glutathione reductase
D. Dopamine beta-hydroxylase
""",
            "answer": """Pantothenic acid is also known as vitamin B5, which is a component of coenzyme A (CoA) and phosphopantetheine. CoA is required for the synthesis and oxidation of fatty acids, and the metabolism of carbohydrates and proteins. One of the enzymes that requires CoA for its function is alpha-ketoglutarate dehydrogenase, a key enzyme in the citric acid cycle (also known as Krebs cycle or TCA cycle).

This enzyme catalyzes the conversion of alpha-ketoglutarate to succinyl CoA, with the concomitant reduction of NAD+ to NADH. Deficiencies in pantothenic acid would impair the function of this enzyme, leading to the observed symptoms.

So, the answer is [B].""",
        },
    ],
}

gpt_chain_of_thoughts_with_ranking = {
    "prompt_name": "gpt_chain_of_thoughts_with_ranking",
    "response_type": "decreasing_order",
    "prompt": few_shots_prompt,
    "examples": [
        {
            "question": """A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow
""",
            "answer": """A supracondylar fracture of the humerus, especially with anterior displacement of the proximal fragment, is a common and serious injury in children. It can cause injury to the brachial artery, leading to compromised blood flow to the forearm and hand. Therefore, the most likely additional finding in this case would be an absent distal radial pulse due to this potential vascular compromise. This is a serious complication and requires immediate orthopedic consultation and likely surgical intervention to prevent long-term damage.

So, the answer is [A].

## Rank the options from most likely to least likely
Answer: [A] > [D] > [B] > [C].""",
        },
        {
            "question": """Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid
""",
            "answer": """This patient's bilateral, symmetric lower extremity swelling extending to the thighs, nonpitting quality, and absence of systemic symptoms (eg, shortness of breath, orthopnea) are suggestive of lymphedema. Lymphedema develops due to a blockage in the lymphatic system that prevents lymph fluid from draining adequately. In this case, the patient's recent surgery (low anterior resection for rectal cancer) probably involved extensive lymph node dissection, leading to secondary lymphedema.

Microscopically, lymph is a protein-rich fluid that also contains lipids, lymphocytes, and other substances. It does not typically contain significant amounts of glycosaminoglycans or hemosiderin.

Therefore, the answer is [C].

## Rank the options from most likely to least likely
Answer: [C] > [D] > [B] > [A].""",
        },
        {
            "question": """A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction
""",
            "answer": """The patient's microcytic, hypochromic anemia and a high reticulocyte count are indicative of increased erythropoiesis, which, in this clinical scenario, is likely due to hemolysis. The positive direct antiglobulin test (Coomb's test) confirms that the hemolysis is immune-mediated. Given that other causes of immune hemolysis such as autoimmune disorders and infections are unlikely in this patient, the most probable cause of hemolysis is a drug-induced immune hemolytic reaction. Drugs can cause immune hemolysis by acting as haptens that bind to the red blood cell membrane, causing the formation of antibodies against the drug-cell complex (immune complex type) or by inducing the formation of autoantibodies against red blood cells (autoimmune type). While any drug can potentially cause immune hemolysis, common culprits include penicillins, cephalosporins, quinidine, and antihypertensives like methyldopa. 

Hence, the answer is [D].

## Rank the options from most likely to least likely
Answer: [D] > [A] > [B] > [C].""",
        },
        {
            "question": """You are counseling a pregnant woman who plans to breast-feed exclusively regarding her newborn's nutritional requirements. The child was born at home and the mother only plans for her newborn to receive vaccinations but no other routine medical care. Which vitamins should be given to the newborn?

A. Folic acid
B. Vitamin K
C. Vitamin D
D. Vitamin K and Vitamin D
""",
            "answer": """The American Academy of Pediatrics (AAP) recommends that all newborns receive a single dose of intramuscular vitamin K at birth. This is because newborns have low stores of vitamin K at birth, leading to a risk of vitamin K deficiency bleeding, also known as hemorrhagic disease of the newborn. This is a potentially life-threatening condition that can cause bleeding into the brain. Because this baby was born at home and the mother plans to avoid routine medical care, this baby may not have received this essential vitamin.

Additionally, the AAP recommends that all infants and children, including those who are breastfed, have a minimum intake of 400 IU/day of vitamin D beginning soon after birth. While breast milk is the best source of most nutrients, it is usually deficient in vitamin D.

The other option, folic acid, is not routinely supplemented in newborns as they usually receive adequate amounts from breast milk or formula.

Therefore, the answer is [D].

## Rank the options from most likely to least likely
Answer: [D] > [C] > [B] > [A].""",
        },
        {
            "question": """An investigator is studying nutritional deficiencies in humans. A group of healthy volunteers are started on a diet deficient in pantothenic acid. After 4 weeks, several of the volunteers develop irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms are fully reversed after reintroduction of pantothenic acid to their diet. The function of which of the following enzymes was most likely impaired in the volunteers during the study?

A. Methionine synthase
B. Alpha-ketoglutarate dehydrogenase
C. Glutathione reductase
D. Dopamine beta-hydroxylase
""",
            "answer": """Pantothenic acid is also known as vitamin B5, which is a component of coenzyme A (CoA) and phosphopantetheine. CoA is required for the synthesis and oxidation of fatty acids, and the metabolism of carbohydrates and proteins. One of the enzymes that requires CoA for its function is alpha-ketoglutarate dehydrogenase, a key enzyme in the citric acid cycle (also known as Krebs cycle or TCA cycle).

This enzyme catalyzes the conversion of alpha-ketoglutarate to succinyl CoA, with the concomitant reduction of NAD+ to NADH. Deficiencies in pantothenic acid would impair the function of this enzyme, leading to the observed symptoms.

So, the answer is [B].

## Rank the options from most likely to least likely
Answer: [B] > [C] > [A] > [D].""",
        },
    ],
}


gpt_chain_of_thoughts_with_ranking_pubmedqa = {
    "prompt_name": "gpt_chain_of_thoughts_with_ranking_pubmedqa",
    "response_type": "decreasing_order",
    "prompt": few_shots_prompt,
    "examples": [
        {
            "question": """Context: Demonstrate that the rabbit may be used in the training of surgery, in addition to present its perioperative care.
Context: Thirty two animals, with age and weight, respectively, from 3 to 5.5 months old and 3000 to 4200 grams, were undergone different periods of pre-operative fasting, exclusive intramuscular anesthesia (ketamine+xylazine), laparotomy with total gastrectomy and total splenectomy. It was dosed the pre-operative (initial) and post-surgical (end) serum blood glucose, in addition to quantify the gastric content after the resection of the part.
Context: The anesthetical-surgical procedure presented a mortality rate of 3.125% (1:32) and a morbidity rate of 6.25% (2:32). It was evidenced an initial mean blood glucose = 199.4 mg/dl and the end = 326.1 mg/dl. In spite of extended fasting (minimum of 2 hours for the absolute fasting and maximum of 8.5 hours for liquids, and 20.5 hours for solids) all animals presented at the end of the surgical procedure any gastric content and a blood glucose increase. Those with fasting for liquids and solids when compared to the quantity of solid gastric content, presented a moderate negative degree of correlation.
Perioperative care in an animal model for training in abdominal surgery: is it necessary a preoperative fasting?

A. no
B. yes
C. maybe
""",
            "answer": """The study showed that despite extended fasting, all animals presented at the end of the surgical procedure with some gastric content and a blood glucose increase. This suggests that preoperative fasting may not be necessary in this animal model for training in abdominal surgery.

Therefore, the answer is [A].

## Rank the options from most likely to least likely
Answer: [A] > [C] > [B].""",
        },
        {
            "question": """Context: As part of a prospective study on quality of life in newly diagnosed lung cancer patients an investigation was carried out to examine whether there were differences among patients' quality of life scores and their socioeconomic status.
Context: Quality of life was measured at two points in time (baseline and three months after initial treatment) using three standard instruments; the Nottingham Health Profile (NHP), the European Organization for Research and Cancer Treatment Quality of Life Questionnaire (EORTC QLQ-C30) and its lung cancer supplement (QLQ-LC13). Socioeconomic status for each individual patient was derived using Carstairs and Morris Deprivation Category ranging from 1 (least deprived) to 7 (most deprived) on the basis of the postcode sector of their address.
Context: In all, 129 lung cancer patients entered into the study. Of these data for 82 patients were complete (at baseline and follow-up). 57% of patients were of lower socioeconomic status and they had more health problems, less functioning, and more symptoms as compared to affluent patients. Of these, physical mobility (P = 0.05), energy (P = 0.01), role functioning (P = 0.04), physical functioning (P = 0.03), and breathlessness (P = 0.02) were significant at baseline. However, at follow-up assessment there was no significant difference between patient groups nor did any consistent pattern emerge.
Quality of life in lung cancer patients: does socioeconomic status matter?

A. yes
B. maybe
C. no
""",
            "answer": """The study found that patients of lower socioeconomic status had more health problems, less functioning, and more symptoms as compared to affluent patients. Therefore, socioeconomic status does matter in the quality of life of lung cancer patients.

So, the answer is [A].

## Rank the options from most likely to least likely
Answer: [A] > [B] > [C].""",
        },
        {
            "question": """Context: Prostate-specific antigen (PSA) levels can show wide fluctuations when repeatedly measured. Here we investigatewd if: (a) biopsy timing influences the prostate cancer (PC) detection rate in patients with fluctuating PSA (flu-PSA) in comparison with patients with steadily increasing PSA (si-PSA); (b) PSA slope estimated in patients with flu-PSA predicts a different risk of cancer detection; (c) flu-PSA and si-PSA patients develop PC in topographically different sites; (d) the behaviour of pre-operative PSA is an expression of a disease with defferent characteristics to the following radical prostatectomy.
Context: The study involved 211 patients who underwent at least a second biopsy after a first negative prostate biopsy. PSA Slope, PSA velocity (PSAV) and PSA doubling time (PSADT) were estimated. Flu-PSA level was defined as a PSA series with at least one PSA value lower than the one immediately preceding it.
Context: 82 patients had flu-PSA levels and 129 si-PSA levels. There were no significant differences between the two groups in terms of cancer detection, clinical or pathological stage, but the si-PSA group with cancer had a higher Gleason score. No difference was found for PSA Slope between flu-PSA patients with cancer and those without.
PSA repeatedly fluctuating levels are reassuring enough to avoid biopsy?

A. maybe
B. no
C. yes
""",
            "answer": """The study found that there were no significant differences between the two groups in terms of cancer detection, clinical or pathological stage, but the si-PSA group with cancer had a higher Gleason score. No difference was found for PSA Slope between flu-PSA patients with cancer and those without. This suggests that fluctuating PSA levels are not necessarily indicative of a lower risk of prostate cancer. Therefore, the answer is [B].

## Rank the options from most likely to least likely
Answer: [B] > [A] > [C].""",
        },
        {
            "question": """Context: To examine whether a history of cancer increased the likelihood of a fall in community-dwelling older adults, and if cancer type, stage, or time since diagnosis increased falls.
Context: A longitudinal, retrospective, cohort study.
Context: A home- and community-based waiver program in Michigan.
Context: 862 older adults aged 65 years or older with cancer compared to 8,617 older adults without cancer using data from the Minimum Data Set-Home Care and Michigan cancer registry.
Context: Reports of falls were examined for 90-180 days. Generalized estimating equations were used to compare differences between the groups.
Context: Cancer, falls, patient characteristics, comorbidities, medications, pain, weight loss, vision, memory recall, and activities, as well as cancer type, stage, and time since diagnosis.
Context: A fall occurred at a rate of 33% in older adults with cancer compared to 29% without cancer (p<0.00). Those with a history of cancer were more likely to fall than those without cancer (adjusted odds ratio 1.16; 95% confidence interval [1.02, 1.33]; p = 0.03). No differences in fall rates were determined by cancer type or stage, and the odds of a fall did not increase when adding time since cancer diagnosis.
Do older adults with cancer fall more often?

A. no
B. maybe
C. yes
""",
            "answer": """The study found that older adults with cancer fell more often than those without cancer. The fall rate was 33% in older adults with cancer compared to 29% without cancer. Those with a history of cancer were more likely to fall than those without cancer (adjusted odds ratio 1.16; 95% confidence interval [1.02, 1.33]; p = 0.03).

So, the answer is [C].

## Rank the options from most likely to least likely
Answer: [C] > [B] > [A].""",
        },
        {
            "question": """Context: The United States Food and Drug Administration implemented federal regulations governing mammography under the Mammography Quality Standards Act (MQSA) of 1992. During 1995, its first year in implementation, we examined the impact of the MQSA on the quality of mammography in North Carolina.
Context: All mammography facilities were inspected during 1993-1994, and again in 1995. Both inspections evaluated mean glandular radiation dose, phantom image evaluation, darkroom fog, and developer temperature. Two mammography health specialists employed by the North Carolina Division of Radiation Protection performed all inspections and collected and codified data.
Context: The percentage of facilities that met quality standards increased from the first inspection to the second inspection. Phantom scores passing rate was 31.6% versus 78.2%; darkroom fog passing rate was 74.3% versus 88.5%; and temperature difference passing rate was 62.4% versus 86.9%.
Has the mammography quality standards act affected the mammography quality in North Carolina?

A. no
B. yes
C. maybe
""",
            "answer": """The percentage of facilities that met quality standards increased from the first inspection to the second inspection. Phantom scores passing rate was 31.6% versus 78.2%; darkroom fog passing rate was 74.3% versus 88.5%; and temperature difference passing rate was 62.4% versus 86.9%. Therefore, the answer is [B].

## Rank the options from most likely to least likely
Answer: [B] > [C] > [A].""",
        },
    ],
}


consider_each_options = {
    "prompt_name": "consider_each_options",
    "response_type": "order",
    "prompt": Template(
        """## Task
1. Start by explaining the relevant facts of the scenario.
2. Discuss each answer option separately, elaborating on their implications.
3. Arrange the options from least likely to most likely.
4. Summarize your findings in the format: "Answer: [Least likely] < [Less likely] < [More likely] < [Most likely]".

{% for item in examples %}## Question
{{ item.question }}
## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
    ),
    "examples": [
        {
            "question": """A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow
""",
            "answer": """Supracondylar fractures are the most common elbow injuries in children. This fracture typically occurs in a fall onto an outstretched hand, as the force is transmitted to the elbow. The distal fragment of the humerus tends to displace and tilt anteriorly, creating a characteristic fracture line.

Let's discuss each of the options individually:

A. Absent distal radial pulse: This could potentially occur if there is severe anterior displacement causing vascular compromise. The brachial artery could become entrapped or occluded, leading to an absent radial pulse. Although this isn't the most common finding, it can be seen in severe cases and requires immediate attention due to the risk of limb ischemia.

B. Radial deviation of the wrist: While fractures can impact alignment, a supracondylar fracture of the humerus wouldn't directly cause radial deviation of the wrist. Radial deviation refers to the movement of the wrist towards the thumb, or the radial bone. This is more likely to be seen in wrist or forearm injuries, rather than elbow fractures.

C. Inability to abduct shoulder: Supracondylar fractures impact the elbow and wouldn't directly impact shoulder movement. The muscles that control shoulder abduction (such as the deltoid and supraspinatus) originate from the scapula and insert on the humerus, above the site of a supracondylar fracture.

D. Inability to flex the elbow: This could certainly occur given the location of the fracture. Pain and swelling may limit motion, but the damage itself wouldn't necessarily cause an inability to flex the elbow, unless there is associated nerve damage. The brachialis, brachioradialis, and biceps brachii muscles that flex the elbow are typically not directly impacted by a supracondylar fracture.

Ranking these options from least likely to most likely, we would have:

Answer: [C] < [B] < [D] < [A].""",
        },
        {
            "question": """Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid
""",
            "answer": """The clinical vignette describes a patient who underwent surgery for rectal cancer and then developed bilateral lower extremity edema extending up to the thighs. This is a common presentation of lymphedema, which can develop after pelvic or retroperitoneal surgery, where the lymph nodes can be disturbed or removed.

Let's evaluate each answer option:

A. Acellular, protein-poor fluid: This is typically found in patients with systemic issues leading to edema, such as in heart failure, liver disease, or kidney disease. This patient's lack of other symptoms such as shortness of breath, jugular venous distension, and his recent surgery make this option less likely.

B. Lymphocytic, hemosiderin-rich fluid: This type of fluid would more likely be found in the setting of chronic venous stasis, such as in venous insufficiency or longstanding varicose veins. Hemosiderin, a breakdown product of hemoglobin, is deposited in tissues when there's chronic venous congestion and extravasation of red blood cells into the interstitial tissue.

C. Lipid-rich, protein-rich fluid: This would be most consistent with lymphedema. Lymph fluid is rich in proteins and lipids, and these substances are deposited in the tissues when lymphatic flow is interrupted, as in this patient who had pelvic surgery.

D. Protein-rich, glycosaminoglycan-rich fluid: This type of fluid can be seen in conditions like myxedema, which is a severe form of hypothyroidism characterized by deposition of glycosaminoglycans in the interstitial space. However, the patient's physical examination did not indicate signs of hypothyroidism.

Ranking the options from least likely to most likely, we get:

Answer: [D] < [A] < [B] < [C].""",
        },
        {
            "question": """A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction
""",
            "answer": """This patient's presentation is suggestive of hemolytic anemia, given her fatigue, pallor, tachycardia, microcytic hypochromic anemia with anisocytosis, and a reticulocyte production index indicative of reticulocytosis. Her positive direct antiglobulin test (DAT, also known as direct Coombs' test) suggests an immune cause for her hemolysis.

Let's evaluate each of the answer choices:

A. Preeclampsia: This is a multisystem disorder of pregnancy characterized by hypertension and either proteinuria or end-organ dysfunction. While some women with severe preeclampsia can have hemolysis, this patient's blood pressure is normal and there is no proteinuria mentioned. Her liver function tests are also normal, which makes preeclampsia less likely.

B. Hereditary spherocytosis: This is a genetic disorder of the red blood cell membrane, leading to red blood cells that are spheroidal in shape, fragile, and prone to hemolysis. However, this type of hemolysis is typically not immune-mediated and would not cause a positive DAT.

C. HELLP syndrome: This syndrome, which stands for Hemolysis, Elevated Liver enzymes, and Low Platelets, is a variant of preeclampsia. However, the patient's liver enzymes and platelet count are normal, and there is no hypertension or proteinuria, making this diagnosis unlikely.

D. Drug-induced immune hemolytic reaction: Certain drugs can induce hemolytic anemia through immune mechanisms. In the case of drug-induced immune hemolytic anemia, the drug binds to the red blood cell surface, inducing an immune response that leads to hemolysis. This patient is taking an antihypertensive drug and has evidence of immune-mediated hemolysis (positive DAT), which makes this diagnosis quite likely.

Ranking the options from least likely to most likely:

Answer: [B] < [C] < [A] < [D].""",
        },
        {
            "question": """You are counseling a pregnant woman who plans to breast-feed exclusively regarding her newborn's nutritional requirements. The child was born at home and the mother only plans for her newborn to receive vaccinations but no other routine medical care. Which vitamins should be given to the newborn?

A. Folic acid
B. Vitamin K
C. Vitamin D
D. Vitamin K and Vitamin D
""",
            "answer": """Breast milk is the recommended source of nutrition for infants because it contains the ideal proportion of nutrients, strengthens the immune system, and promotes bonding. However, it lacks adequate amounts of two important vitamins: vitamin K and vitamin D.

Let's discuss each option separately:

A. Folic acid: While folic acid is crucial for cellular division and differentiation, particularly during embryogenesis, breast milk usually contains enough folic acid to meet an infant's needs, making supplementation typically unnecessary.

B. Vitamin K: This is a vital factor in the coagulation cascade. Newborns have low stores of vitamin K, which puts them at risk of vitamin K deficiency bleeding. Thus, a single intramuscular dose of vitamin K is routinely given to all newborns immediately after birth. In this case, the newborn was born at home and likely did not receive this routine dose of vitamin K, which is important to prevent hemorrhagic disease of the newborn.

C. Vitamin D: Breast milk contains insufficient amounts of vitamin D, which is crucial for bone health and growth. Without supplementation, infants who are exclusively breastfed are at risk of developing vitamin D deficiency and rickets. The American Academy of Pediatrics recommends all breastfed infants receive 400 IU/day of oral vitamin D supplementation, starting within the first few days of life.

D. Vitamin K and Vitamin D: As explained above, these are the two important vitamins that a breastfed infant is likely to lack, and both should be supplemented.

Therefore, ranking from the least to most likely needed vitamins, the order is:

Answer: [A] < [B] < [C] < [D].""",
        },
        {
            "question": """An investigator is studying nutritional deficiencies in humans. A group of healthy volunteers are started on a diet deficient in pantothenic acid. After 4 weeks, several of the volunteers develop irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms are fully reversed after reintroduction of pantothenic acid to their diet. The function of which of the following enzymes was most likely impaired in the volunteers during the study?

A. Methionine synthase
B. Alpha-ketoglutarate dehydrogenase
C. Glutathione reductase
D. Dopamine beta-hydroxylase
""",
            "answer": """Pantothenic acid, also known as vitamin B5, is a crucial component of coenzyme A (CoA) and also plays a role in the synthesis of the phosphopantetheine moiety of the acyl carrier protein. CoA is involved in many metabolic pathways, including the TCA cycle and fatty acid metabolism.

Let's examine each option separately:

A. Methionine synthase: This enzyme is dependent on vitamin B12 (cobalamin) and is involved in the conversion of homocysteine to methionine. It is not directly associated with pantothenic acid function.

B. Alpha-ketoglutarate dehydrogenase: This is a key enzyme in the TCA (Krebs) cycle that catalyzes the conversion of alpha-ketoglutarate to succinyl-CoA. This reaction is a multi-step process that involves the decarboxylation of alpha-ketoglutarate, the transfer of electrons to NAD+, and the subsequent transfer of the remaining acyl group to CoA, generating succinyl-CoA. Because CoA is derived from pantothenic acid, a deficiency of pantothenic acid could impair the function of this enzyme.

C. Glutathione reductase: This enzyme is involved in maintaining the reduced form of glutathione in the body, which helps protect cells from oxidative damage. It requires flavin adenine dinucleotide (FAD), a coenzyme derived from riboflavin (vitamin B2), not pantothenic acid.

D. Dopamine beta-hydroxylase: This enzyme is involved in the biosynthesis of norepinephrine from dopamine. It requires vitamin C (ascorbic acid) and copper for its activity, not pantothenic acid.

So, ranking these options from least likely to most likely to be impaired due to pantothenic acid deficiency, we have:

Answer: [A] < [C] < [D] < [B].""",
        },
    ],
}

consider_each_options_without_judgement = {
    "prompt_name": "consider_each_options_without_judgement",
    "response_type": "order",
    "prompt": Template(
        """## Question
A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow

## Discussion of each option without mentioning if it is correct or not
A. The proposed answer suggests that the 6-year-old girl with a supracondylar fracture of the humerus and anterior displacement of the proximal fragment may present with an absent distal radial pulse. This finding is related to the potential injury to the blood vessels in the area of the fracture, specifically the brachial artery. The brachial artery is responsible for supplying blood to the forearm, and disruption of its function could lead to an absent or weak distal radial pulse. In this scenario, it is important to consider the possibility of vascular compromise and monitor the patient's distal circulation carefully.

B. The proposed answer suggests that further evaluation of the 6-year-old girl with a supracondylar fracture of the humerus and anterior displacement of the proximal fragment is most likely to show radial deviation of the wrist. Radial deviation refers to the movement of the wrist towards the thumb side, which is the radial side of the forearm. This can occur due to various reasons, including fractures, injuries, or other pathological conditions affecting the bones, ligaments, or muscles of the forearm and wrist. In this case, the proposed answer is focusing on the potential consequences of the supracondylar fracture on the alignment and function of the wrist joint.

C. The proposed answer suggests that the 6-year-old girl with a supracondylar fracture of the humerus and anterior displacement of the proximal fragment would most likely exhibit an inability to abduct her shoulder. The abduction of the shoulder involves the movement of the arm away from the body, and it primarily engages the muscles around the shoulder joint. In this case, the focus is on the elbow injury and its associated symptoms, such as bruising, swelling, tenderness, and limited range of motion due to pain. The relationship between the elbow fracture and shoulder abduction should be further explored to determine the accuracy of the proposed answer.

D. A supracondylar fracture of the humerus with anterior displacement of the proximal fragment is a common injury in children, often resulting from a fall on an outstretched arm. This type of fracture can lead to various complications and associated findings. The proposed answer suggests that the child may experience an inability to flex the elbow. This could be due to the involvement of the surrounding muscles, ligaments, and nerves, as well as the direct impact of the displaced bone fragment on the joint's normal functioning. Pain, swelling, and limited range of motion are common symptoms that may contribute to difficulty in flexing the elbow.

## Discuss the relevant facts and arrange the options
In this scenario, a 6-year-old girl has a supracondylar fracture of the humerus with anterior displacement of the proximal fragment after falling onto her outstretched right arm. The relevant facts include swelling and pain in the right elbow, limited range of motion, and the presence of the fracture as seen on the x-ray. We will now arrange the answer options from least likely to most likely based on these facts:

Inability to abduct shoulder (C): This is the least likely finding because shoulder abduction is primarily related to the shoulder joint and its surrounding muscles, rather than the elbow joint. The scenario does not provide any direct evidence linking the injury to the shoulder abduction.

Radial deviation of the wrist (B): This is a less likely finding, as radial deviation of the wrist is primarily related to the wrist joint and its surrounding structures. While a fracture in the elbow region could potentially affect the wrist joint's alignment and function, the direct impact of this specific fracture on radial deviation is not clear.

Inability to flex the elbow (D): This is a more likely finding due to the involvement of the surrounding muscles, ligaments, and nerves, as well as the direct impact of the displaced bone fragment on the joint's normal functioning. Pain, swelling, and limited range of motion are common symptoms that may contribute to difficulty in flexing the elbow.

Absent distal radial pulse (A): This is the most likely finding, as the fracture's location and the anterior displacement of the proximal fragment could result in injury to the brachial artery, leading to an absent or weak distal radial pulse. Vascular compromise is an important consideration in this scenario and should be monitored closely.

Answer: [C] < [B] < [D] < [A]

## Question
Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid

## Discussion of each option without mentioning if it is correct or not
A. The proposed answer suggests that microscopic examination of the interstitial space in the patient's lower extremities would most likely show acellular, protein-poor fluid. This type of fluid is typically found in non-inflammatory edema, which is caused by an imbalance in the hydrostatic and oncotic forces within the capillaries. This could result from conditions such as venous insufficiency, heart failure, or nephrotic syndrome. However, the patient's clinical presentation does not specifically indicate this type of fluid accumulation.

B. The proposed answer suggests that microscopic examination of the interstitial space in the patient's lower extremities would most likely show lymphocytic, hemosiderin-rich fluid. This type of fluid is characteristic of chronic venous insufficiency, where long-standing venous hypertension leads to the breakdown of red blood cells and the deposition of hemosiderin. This can cause the skin to become discolored and inflamed, a condition known as stasis dermatitis. However, the patient's clinical presentation does not specifically indicate this type of fluid accumulation.

C. The proposed answer suggests that microscopic examination of the interstitial space in the patient's lower extremities would most likely show lipid-rich, protein-rich fluid. This type of fluid is characteristic of lipedema, a chronic disorder characterized by the abnormal deposition of adipose tissue and fluid in the subcutaneous layer of the lower extremities. It is often accompanied by pain, tenderness, and easy bruising. However, the patient's clinical presentation does not specifically indicate this type of fluid accumulation.

D. The proposed answer suggests that microscopic examination of the interstitial space in the patient's lower extremities would most likely show protein-rich, glycosaminoglycan-rich fluid. This type of fluid is characteristic of lymphedema, a condition in which lymphatic fluid accumulates in the interstitial tissue due to impaired lymphatic drainage. Lymphedema can be primary (congenital) or secondary (acquired), with the latter often resulting from surgery, radiation therapy, or trauma. In this scenario, the patient's recent surgery for rectal cancer may have disrupted the lymphatic drainage, leading to the development of lymphedema.

## Discuss the relevant facts and arrange the options
The relevant facts in this scenario include the patient's recent low anterior resection for rectal cancer, the absence of fever, chills, or shortness of breath, and the presence of bilateral non-pitting edema in the lower extremities. Based on these facts, we can arrange the answer options from least likely to most likely:

Acellular, protein-poor fluid (A): This is the least likely option, as the patient's clinical presentation does not specifically indicate non-inflammatory edema.

Lymphocytic, hemosiderin-rich fluid (B): This is less likely because the patient's clinical presentation does not specifically indicate chronic venous insufficiency or stasis dermatitis.

Protein-rich, glycosaminoglycan-rich fluid (D): This is also less likely because the patient's clinical presentation does not specifically indicate lymphedema.

Lipid-rich, protein-rich fluid (C): This is the most likely option, as the patient's recent surgery for rectal cancer and the presence of bilateral non-pitting edema in the lower extremities could be indicative of a condition like lipedema.

Answer: [A] < [B] < [D] < [C]

## Question
A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction

## Discussion of each option without mentioning if it is correct or not
A. The proposed answer suggests that the most likely diagnosis in this patient is preeclampsia. Preeclampsia is a pregnancy complication characterized by high blood pressure and signs of damage to another organ system, often the liver and kidneys. While the patient has a history of gestational hypertension, her current blood pressure is within the normal range, and her liver and kidney function tests are normal. Additionally, her anemia and positive direct antiglobulin test do not fit the typical presentation of preeclampsia.

B. The proposed answer suggests that the most likely diagnosis in this patient is hereditary spherocytosis. Hereditary spherocytosis is an inherited disorder that affects the red blood cells, causing them to adopt a spherical shape and break down prematurely, leading to hemolytic anemia. While the patient's microcytic, hypochromic anemia and positive direct antiglobulin test may suggest hemolytic anemia, the presence of anisocytosis and her clinical presentation do not specifically point to hereditary spherocytosis.

C. The proposed answer suggests that the most likely diagnosis in this patient is HELLP syndrome. HELLP syndrome is a life-threatening pregnancy complication that stands for Hemolysis, Elevated Liver enzymes, and Low Platelet count. Although the patient has hemolytic anemia, her liver function tests and coagulation studies are normal, making HELLP syndrome less likely.

D. The proposed answer suggests that the most likely diagnosis in this patient is drug-induced immune hemolytic reaction. This condition occurs when a medication triggers the immune system to attack the body's red blood cells, leading to hemolytic anemia. The patient's positive direct antiglobulin test, anemia, and recent initiation of an antihypertensive medication are consistent with this diagnosis.

## Discuss the relevant facts and arrange the options
The relevant facts in this scenario include the patient's gestational hypertension, recent initiation of an antihypertensive medication, anemia, positive direct antiglobulin test, and normal liver and kidney function tests. Based on these facts, we can arrange the answer options from least likely to most likely:

Preeclampsia (A): This is the least likely diagnosis, as the patient's blood pressure is within the normal range, and her clinical presentation does not fit the typical presentation of preeclampsia.

Hereditary spherocytosis (B): This is less likely because the patient's clinical presentation does not specifically point to hereditary spherocytosis.

HELLP syndrome (C): This is also less likely because the patient's liver function tests and coagulation studies are normal, making HELLP syndrome less likely.

Drug-induced immune hemolytic reaction (D): This is the most likely diagnosis, as the patient's positive direct antiglobulin test, anemia, and recent initiation of an antihypertensive medication are consistent with this condition.

Answer: [A] < [B] < [C] < [D]

## Question
{{ question }}
## Discussion of each option without mentioning if it is correct or not
"""
    ),
}

consider_each_options_independently = {
    "prompt_name": "consider_each_options_independently",
    "response_type": "order",
    "prompt": Template(
        """## Task
1. Begin by individually evaluating each answer option.
2. Discuss the pertinent facts from the scenario that help in narrowing down the options.
3. Organize the options in order of likelihood, from least likely to most likely.
4. Summarize your conclusions using the format: "Answer: [Least likely] < [Less likely] < [More likely] < [Most likely]".

## Question
A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow

## Assess each answer option
A. The proposed answer of an absent distal radial pulse is likely correct. A supracondylar fracture of the humerus with anterior displacement of the proximal fragment is a common pediatric injury that can result in neurovascular compromise. The most concerning complication of this type of fracture is damage to the brachial artery, which can lead to an absent or weak distal radial pulse. It is essential to assess the neurovascular status of the patient, including checking for distal pulses, capillary refill, and sensation, as any signs of compromise warrant urgent intervention to prevent long-term complications.

B. The proposed answer of radial deviation of the wrist is incorrect. In a supracondylar fracture of the humerus with anterior displacement of the proximal fragment, the most likely finding would be damage to the brachial artery, which could result in compromised blood flow to the forearm and hand. This is because the brachial artery runs close to the elbow joint and can be injured when the proximal fragment of the humerus is displaced anteriorly. Additionally, nerve injuries, particularly to the median and radial nerves, may also be observed in such fractures.

C. The proposed answer of "inability to abduct shoulder" may not be the most likely finding in this case. A supracondylar fracture of the humerus with anterior displacement of the proximal fragment is more likely to be associated with neurovascular complications, such as injury to the brachial artery or median nerve. This could result in findings such as weak or absent radial pulse, pallor, and decreased sensation in the hand. While shoulder abduction may be affected due to pain and discomfort, it is not the primary concern in this specific type of fracture.

D. The proposed answer, "inability to flex the elbow," is a likely finding in this case. A supracondylar fracture of the humerus with anterior displacement of the proximal fragment can result in damage to the surrounding structures, including nerves and blood vessels. This may lead to difficulty in moving the elbow joint, particularly flexion, due to pain and swelling. Additionally, there may be sensory and motor deficits in the forearm and hand, depending on the specific nerves affected. Prompt evaluation and treatment are necessary to minimize the risk of complications and ensure proper healing of the fracture.

## Discuss the relevant facts, explain and arrange the options
Relevant facts include that the patient is a 6-year-old girl who has suffered a supracondylar fracture of the humerus with anterior displacement of the proximal fragment following a fall. The primary concern in this type of injury is neurovascular compromise, which can be caused by damage to the brachial artery or nerves.

Based on the assessment of the answer options, the least likely finding is radial deviation of the wrist (B), as it does not directly relate to the neurovascular complications associated with a supracondylar fracture. The less likely finding is the inability to abduct the shoulder (C), as it may be affected due to pain but is not the primary concern in this specific type of fracture. The more likely finding is the inability to flex the elbow (D), as damage to surrounding structures can cause difficulty in moving the elbow joint. The most likely finding is an absent distal radial pulse (A), which can result from damage to the brachial artery and is a primary concern in this type of injury.

Answer: [B] < [C] < [D] < [A]

# Question
Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid

## Assess each answer option
A. The patient's symptoms of bilateral non-pitting edema without fever, chills, or shortness of breath suggest a possible postoperative complication or lymphatic obstruction. The absence of erythema or rash indicates that it is not an inflammatory process. In this case, the microscopic examination of the interstitial space in the patient's lower extremities would most likely show the presence of acellular, protein-poor fluid. This finding would be consistent with lymphatic obstruction or impaired venous return, which can occur as a result of surgery or other factors affecting the lymphatic system or venous circulation in the lower extremities.

B. The patient's symptoms of bilateral non-pitting edema without fever, chills, or shortness of breath, along with the absence of jugular venous distention and erythema or rash, suggest that the swelling may be related to lymphatic obstruction or insufficiency. This could be a result of the recent surgery, which may have disrupted lymphatic vessels. Microscopic examination of the interstitial space in the patient's lower extremities would most likely show the presence of lymphocytic, hemosiderin-rich fluid, as this is indicative of lymphatic congestion and the accumulation of fluid and cellular debris in the affected tissues.

C. The patient's symptoms of bilateral non-pitting edema in the lower extremities without fever, chills, or shortness of breath suggest a possible lymphatic obstruction or impaired lymphatic drainage. This could be a postoperative complication related to his recent surgery for rectal cancer. The presence of deep flexion creases further supports this possibility. Microscopic examination of the interstitial space in this patient's lower extremities would most likely show the presence of lipid-rich, protein-rich fluid, which is characteristic of lymphatic fluid. Lymphatic obstruction or impairment should be further assessed and managed by the patient's physician.

D. Your proposed answer is correct. The patient's symptoms of bilateral non-pitting edema without fever, chills, or shortness of breath suggest a fluid accumulation in the interstitial space of his lower extremities. Microscopic examination of the interstitial space in this patient's lower extremities would most likely show the presence of protein-rich, glycosaminoglycan-rich fluid. This fluid accumulation can be a result of various factors, such as venous insufficiency, lymphatic obstruction, or an increase in capillary permeability, among others.

## Discuss the relevant facts, explain and arrange the options
Relevant facts include that the patient is a 52-year-old man who underwent low anterior resection for rectal cancer two weeks ago and is now experiencing bilateral non-pitting edema in both feet without fever, chills, or shortness of breath. The absence of erythema or rash and the presence of deep flexion creases suggest a possible lymphatic obstruction or impaired lymphatic drainage, which could be a postoperative complication related to his recent surgery.

Based on the assessment of the answer options, the least likely finding in the microscopic examination of the interstitial space in the patient's lower extremities is acellular, protein-poor fluid (A), as it does not account for the possible lymphatic obstruction or impairment. The less likely finding is lymphocytic, hemosiderin-rich fluid (B), as it is indicative of lymphatic congestion but does not fully explain the non-pitting nature of the edema. The more likely finding is protein-rich, glycosaminoglycan-rich fluid (D), which can result from various factors, including venous insufficiency and increased capillary permeability. The most likely finding is lipid-rich, protein-rich fluid (C), which is characteristic of lymphatic fluid and consistent with the patient's symptoms, suggesting lymphatic obstruction or impairment as the cause of the edema.

Answer: [A] < [B] < [D] < [C]

## Question
A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction

## Assess each answer option
A. The most likely diagnosis in this patient is not preeclampsia. The symptoms and laboratory results, including microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, and a positive direct antiglobulin test, are more indicative of autoimmune hemolytic anemia. Preeclampsia is characterized by hypertension and proteinuria, which are not mentioned in this case. Additionally, the patient's blood pressure is controlled with medication, and other tests, such as LFTs, creatinine, and urinalysis, are normal, making preeclampsia less likely.

B. The patient's presentation and laboratory results suggest she is suffering from anemia, particularly with the findings of microcytic, hypochromic anemia, anisocytosis, and a positive direct antiglobulin test. Although hereditary spherocytosis is a possibility, the presence of a positive direct antiglobulin test makes autoimmune hemolytic anemia a more likely diagnosis. In autoimmune hemolytic anemia, the immune system mistakenly attacks and destroys red blood cells, leading to anemia. The positive direct antiglobulin test indicates the presence of antibodies on the surface of red blood cells, which is consistent with this diagnosis.

C. The most likely diagnosis in this patient is not HELLP syndrome. HELLP syndrome typically presents with hemolysis, elevated liver enzymes, and low platelet count, which is not consistent with the patient's laboratory results. The positive direct antiglobulin test and the findings of microcytic, hypochromic anemia with anisocytosis suggest that she may have autoimmune hemolytic anemia, which can cause fatigue and pallor. It is important for the healthcare provider to consider this diagnosis and manage the patient accordingly.

D. The most likely diagnosis in this patient is drug-induced immune hemolytic anemia. The patient's presentation of fatigue, pallor, tachycardia, microcytic and hypochromic anemia with anisocytosis, and a positive direct antiglobulin test suggests an immune-mediated hemolytic process. The patient's history of taking an antihypertensive drug since week 24 of gestation may have triggered this immune response, leading to the destruction of red blood cells and anemia. The normal levels of LFTs, creatinine, ferritin, vitamin B12, coagulation studies, and urinalysis further support this diagnosis by ruling out other potential causes of anemia.

## Discuss the relevant facts, explain and arrange the options
Relevant facts include the patient's presentation of fatigue, pallor, tachycardia, and laboratory results of microcytic, hypochromic anemia with anisocytosis, a positive direct antiglobulin test, and normal LFTs, creatinine, ferritin, vitamin B12, coagulation studies, and urinalysis. Additionally, the patient has been taking an antihypertensive drug since week 24 of gestation.

Based on the assessment of the answer options, the least likely diagnosis is preeclampsia (A), as the symptoms and laboratory results are more indicative of autoimmune hemolytic anemia, and the patient's blood pressure is controlled with medication. The less likely diagnosis is hereditary spherocytosis (B), as the positive direct antiglobulin test makes autoimmune hemolytic anemia more likely. The more likely diagnosis is HELLP syndrome (C), but the patient's laboratory results do not fully align with this condition, making drug-induced immune hemolytic anemia (D) the most likely diagnosis. The patient's history of taking an antihypertensive drug and the presence of a positive direct antiglobulin test suggest an immune-mediated hemolytic process triggered by the drug.

Answer: [A] < [B] < [C] < [D]

## Question
{{ question }}
## Assess each answer option
A. {{ assessments.A }}

B. {{ assessments.B }}

C. {{ assessments.C }}

D. {{ assessments.D }}

## Discuss the relevant facts, explain and arrange the options
"""
    ),
    "assessment": Template(
        """## Question
{{ question }}

# Proposed answer
{{ option }}

# Assessment (100 words max)
"""
    ),
}

score_each_option = {
    "prompt_name": "score_each_option",
    "response_type": "probs",
    "prompt": Template(
        """## Task
1. Start by explaining the relevant facts of the scenario.
2. Discuss each answer option separately, estimating the correct probability of each option.
3. Summarize your findings in the format: "Probability: A = [??]%, B = [??]%, C = [??]%, D = [??]%".

{% for item in examples %}## Question
{{ item.question }}
## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
    ),
    "examples": [
        {
            "question": """A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow
""",
            "answer": """In this scenario, a 6-year-old girl has fallen onto her outstretched right arm, resulting in swelling, pain, and limited range of motion in the right elbow. The x-ray shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. We will now examine each answer option and estimate the probability of each being correct.

A. Absent distal radial pulse: A supracondylar fracture can potentially damage the blood vessels around the elbow, leading to a compromised or absent distal radial pulse. This is a plausible consequence of this type of injury. Probability: A = 80%

B. Radial deviation of the wrist: Radial deviation of the wrist refers to the wrist bending toward the thumb side. Although this could occur with some injuries, it is less likely to be directly related to a supracondylar fracture of the humerus. Probability: B = 10%

C. Inability to abduct shoulder: Abduction of the shoulder primarily involves the shoulder joint rather than the elbow. While the patient may be in pain and reluctant to move the entire arm, the supracondylar fracture itself is not directly responsible for an inability to abduct the shoulder. Probability: C = 5%

D. Inability to flex the elbow: Considering the swelling, pain, and limited range of motion observed in the patient's right elbow, it is likely that the patient will have difficulty flexing the elbow due to the fracture. Probability: D = 75%

In summary, the estimated probabilities for each option are as follows:
Probability: A = 80%, B = 10%, C = 5%, D = 75%.""",
        },
        {
            "question": """Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid
""",
            "answer": """In this scenario, a 52-year-old man underwent a low anterior resection for rectal cancer and presented with bilateral non-pitting edema in both feet two weeks after surgery. We will now examine each answer option and estimate the probability of each being correct.

A. Acellular, protein-poor fluid: Non-pitting edema is typically associated with lymphedema, which is characterized by a high protein content in the interstitial fluid. Therefore, it is less likely to be an acellular, protein-poor fluid. Probability: A = 5%

B. Lymphocytic, hemosiderin-rich fluid: Hemosiderin-rich fluid is more suggestive of venous insufficiency, which usually results in pitting edema. Since the patient has non-pitting edema, this option is less likely. Probability: B = 10%

C. Lipid-rich, protein-rich fluid: Non-pitting edema can be seen in conditions such as lipedema, which is characterized by the accumulation of lipid-rich and protein-rich fluid in the interstitial space. Given the patient's presentation of bilateral non-pitting edema, this option has the highest probability. Probability: C = 70%

D. Protein-rich, glycosaminoglycan-rich fluid: Although non-pitting edema is commonly associated with lymphedema, which is characterized by the accumulation of protein-rich fluid in the interstitial space, the absence of other lymphedema-related signs (such as erythema or rash) makes this option less likely. Probability: D = 15%

In summary, the estimated probabilities for each option are as follows:
Probability: A = 5%, B = 10%, C = 70%, D = 15%.""",
        },
        {
            "question": """A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction
""",
            "answer": """In this scenario, a 34-year-old pregnant woman presents with increased fatigue, generalized pallor, and laboratory results indicating microcytic, hypochromic anemia with anisocytosis, a positive direct antiglobulin test, and a reticulocyte production index of 3.1. We will now examine each answer option and estimate the probability of each being correct.

A. Preeclampsia: Although the patient has a history of gestational hypertension, her current blood pressure is normal, and there is no mention of proteinuria or other signs of preeclampsia. Probability: A = 5%

B. Hereditary spherocytosis: This condition is characterized by spherically shaped red blood cells and can cause hemolytic anemia. However, the patient's anemia is described as microcytic and hypochromic, which is not typical of hereditary spherocytosis. Probability: B = 10%

C. HELLP syndrome: HELLP syndrome is a severe form of preeclampsia, characterized by hemolysis, elevated liver enzymes, and low platelet count. The patient does have some signs of hemolysis, but her liver function tests and coagulation studies are normal, making this less likely. Probability: C = 15%

D. Drug-induced immune hemolytic reaction: The patient's positive direct antiglobulin test suggests an immune-mediated hemolytic process. Given her history of taking an antihypertensive drug, it is possible that her anemia is a result of a drug-induced immune hemolytic reaction. Probability: D = 70%

In summary, the estimated probabilities for each option are as follows:
Probability: A = 5%, B = 10%, C = 15%, D = 70%.""",
        },
        {
            "question": """You are counseling a pregnant woman who plans to breast-feed exclusively regarding her newborn's nutritional requirements. The child was born at home and the mother only plans for her newborn to receive vaccinations but no other routine medical care. Which vitamins should be given to the newborn?

A. Folic acid
B. Vitamin K
C. Vitamin D
D. Vitamin K and Vitamin D
""",
            "answer": """In the scenario presented, the newborn will be breastfed exclusively and will not receive any routine medical care besides vaccinations. We will now consider which vitamins should be given to the newborn.

A. Folic acid: Folic acid is essential for pregnant women to prevent neural tube defects in the developing fetus. However, it is not routinely given to newborns as a supplement. Probability: A = 5%

B. Vitamin K: Vitamin K is crucial for blood clotting and preventing bleeding in newborns. Newborns have low stores of vitamin K, and breast milk does not provide an adequate amount. Vitamin K is typically given as an injection shortly after birth, especially if the child will not receive routine medical care. Probability: B = 40%

C. Vitamin D: Exclusively breastfed infants are at risk of vitamin D deficiency, as breast milk does not provide enough vitamin D to meet their needs. Vitamin D supplementation is recommended for breastfed infants to ensure proper bone development and prevent rickets. Probability: C = 25%

D. Vitamin K and Vitamin D: Given that newborns have low stores of vitamin K and breast milk does not provide enough vitamin D, it is essential to provide both vitamin K and vitamin D supplements to ensure proper blood clotting, bone development, and overall health. Probability: D = 70%

In summary, the estimated probabilities for each option are as follows:
Probability: A = 5%, B = 40%, C = 25%, D = 70%.""",
        },
        {
            "question": """An investigator is studying nutritional deficiencies in humans. A group of healthy volunteers are started on a diet deficient in pantothenic acid. After 4 weeks, several of the volunteers develop irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms are fully reversed after reintroduction of pantothenic acid to their diet. The function of which of the following enzymes was most likely impaired in the volunteers during the study?

A. Methionine synthase
B. Alpha-ketoglutarate dehydrogenase
C. Glutathione reductase
D. Dopamine beta-hydroxylase
""",
            "answer": """In this scenario, healthy volunteers were started on a diet deficient in pantothenic acid, which led to the development of irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms were reversed after reintroducing pantothenic acid to their diet. We will now consider which enzyme's function was most likely impaired during the study.

A. Methionine synthase: This enzyme is involved in the conversion of homocysteine to methionine and requires vitamin B12 as a cofactor. Pantothenic acid deficiency would not directly affect this enzyme's function. Probability: A = 5%

B. Alpha-ketoglutarate dehydrogenase: This enzyme is involved in the citric acid cycle and requires several cofactors, including thiamine pyrophosphate, NAD+, FAD, and coenzyme A (CoA). Pantothenic acid is an essential component of CoA, so a deficiency in pantothenic acid would impair the function of alpha-ketoglutarate dehydrogenase. Probability: B = 80%

C. Glutathione reductase: This enzyme is involved in maintaining cellular redox balance and requires FAD as a cofactor. Pantothenic acid deficiency would not directly impair this enzyme's function. Probability: C = 5%

D. Dopamine beta-hydroxylase: This enzyme is involved in the synthesis of norepinephrine from dopamine and requires vitamin C as a cofactor. Pantothenic acid deficiency would not directly affect this enzyme's function. Probability: D = 10%

In summary, the estimated probabilities for each option are as follows:
Probability: A = 5%, B = 80%, C = 5%, D = 10%.""",
        },
    ],
}

gpt_chain_of_thoughts_with_context = {
    "prompt_name": "gpt_chain_of_thoughts_with_context",
    "response_type": "answer_letter",
    "prompt": few_shots_prompt,
    "examples": [
        {
            "question": """A 6-year-old girl is brought to the emergency department because of right elbow swelling and pain 30 minutes after falling onto her outstretched right arm. She has been unable to move her right elbow since the fall. Physical examination shows bruising, swelling, and tenderness of the right elbow; range of motion is limited by pain. An x-ray of the right arm shows a supracondylar fracture of the humerus with anterior displacement of the proximal fragment. Further evaluation is most likely to show which of the following findings?

A. Absent distal radial pulse
B. Radial deviation of the wrist
C. Inability to abduct shoulder
D. Inability to flex the elbow
""",
            "answer": """Considering potential complications, the most severe is injury to the brachial artery. The fracture can compromise this artery, affecting blood flow to the forearm and hand. This can result in an absent distal radial pulse.
Answer: A""",
        },
        {
            "question": """Two weeks after undergoing low anterior resection for rectal cancer, a 52-year-old man comes to the physician because of swelling in both feet. He has not had any fever, chills, or shortness of breath. His temperature is 36°C (96.8°F) and pulse is 88/min. Physical examination shows a normal thyroid and no jugular venous distention. Examination of the lower extremities shows bilateral non-pitting edema that extends from the feet to the lower thigh, with deep flexion creases. His skin is warm and dry, and there is no erythema or rash. Microscopic examination of the interstitial space in this patient's lower extremities would be most likely to show the presence of which of the following?

A. Acellular, protein-poor fluid
B. Lymphocytic, hemosiderin-rich fluid
C. Lipid-rich, protein-rich fluid
D. Protein-rich, glycosaminoglycan-rich fluid
""",
            "answer": """The clinical presentation is most consistent with lymphedema, which commonly occurs secondary to lymphatic obstruction or damage, such as from pelvic lymph node dissection during his recent surgery for rectal cancer. Lymphedema is characterized by an accumulation of protein-rich lymphatic fluid in the interstitial spaces. Lipids are also a significant component of lymph. Therefore, the most likely finding in this patient's lower extremities would be lipid-rich, protein-rich fluid.
Answer: C""",
        },
        {
            "question": """A 34-year-old gravida 2, para 1 woman at 37+6 weeks of gestation presents for elective cesarean delivery. She says she has been having increased fatigue over the past few weeks. Past medical history includes gestational hypertension for which she has been taking an antihypertensive drug twice daily since week 24. Her vital signs include: temperature 36.7°C (98.0°F), blood pressure 120/75 mm Hg, pulse 127/min. Physical examination reveals generalized pallor. Her laboratory results reveal microcytic, hypochromic anemia with anisocytosis, hemoglobin of 9 g/dL, a differential with 14% lymphocytes, an ESR of 22 mm/hr, and a reticulocyte production index of 3.1. A direct antiglobulin test is positive. LFTs, creatinine, ferritin level, vitamin B12 level, coagulation studies, and urinalysis are normal. Which of the following is the most likely diagnosis in this patient?

A. Preeclampsia
B. Hereditary spherocytosis
C. HELLP syndrome
D. Drug-induced immune hemolytic reaction
""",
            "answer": """Considering the patient's clinical presentation and laboratory findings, it's important to note that her current antihypertensive medication could be playing a role in her condition. Certain drugs can cause immune-mediated hemolytic anemia by triggering the immune system to produce antibodies against red blood cells. This process often leads to premature destruction of red blood cells, resulting in anemia. Given that the woman has been on an antihypertensive drug, which is known to occasionally cause such reactions, and considering her positive direct antiglobulin test and symptoms, the most likely diagnosis is a drug-induced immune hemolytic reaction.
Answer: D""",
        },
        {
            "question": """You are counseling a pregnant woman who plans to breast-feed exclusively regarding her newborn's nutritional requirements. The child was born at home and the mother only plans for her newborn to receive vaccinations but no other routine medical care. Which vitamins should be given to the newborn?

A. Folic acid
B. Vitamin K
C. Vitamin D
D. Vitamin K and Vitamin D
""",
            "answer": """Given the newborn is exclusively breastfed and receiving minimal medical care, two vitamins are crucial. Vitamin K is needed immediately after birth to prevent bleeding. Vitamin D supplementation is necessary as breast milk typically doesn't provide sufficient amounts.
Answer: D""",
        },
        {
            "question": """An investigator is studying nutritional deficiencies in humans. A group of healthy volunteers are started on a diet deficient in pantothenic acid. After 4 weeks, several of the volunteers develop irritability, abdominal cramps, and burning paresthesias of their feet. These symptoms are fully reversed after reintroduction of pantothenic acid to their diet. The function of which of the following enzymes was most likely impaired in the volunteers during the study?

A. Methionine synthase
B. Alpha-ketoglutarate dehydrogenase
C. Glutathione reductase
D. Dopamine beta-hydroxylase
""",
            "answer": """Alpha-ketoglutarate dehydrogenase is a key enzyme in the citric acid cycle (Krebs cycle). It catalyzes the conversion of alpha-ketoglutarate to succinyl-CoA, a reaction that involves CoA, a molecule derived from pantothenic acid. So, deficiency in pantothenic acid would impair the function of alpha-ketoglutarate dehydrogenase. Therefore, the most likely impaired enzyme in the volunteers during the study was alpha-ketoglutarate dehydrogenase.
Answer: B""",
        },
    ],
}

few_shots_cot_prompt = Template(
    """{% for item in examples %}## Question
{{ item.question }}
## Explanation
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Explanation
"""
)

gpt_chain_of_thoughts_enough_or_not = {
    "prompt_name": "gpt_chain_of_thoughts_enough_or_not",
    "response_type": "answer_letter_md",
    "prompt": few_shots_cot_prompt,
    "examples": [
        {
            "question": """Context: Lymphedema may be identified by simpler circumference changes as compared with changes in limb volume.
Context: Ninety breast cancer patients were prospectively enrolled in an academic trial, and seven upper extremity circumferences were measured quarterly for 3 years. A 10% volume increase or greater than 1 cm increase in arm circumference identified lymphedema with verification by a lymphedema specialist. Sensitivity and specificity of several different criteria for detecting lymphedema were compared using the academic trial as the standard.
Context: Thirty-nine cases of lymphedema were identified by the academic trial. Using a 10% increase in circumference at two sites as the criterion, half the lymphedema cases were detected (sensitivity 37%). When using a 10% increase in circumference at any site, 74.4% of cases were detected (sensitivity 49%). Detection by a 5% increase in circumference at any site was 91% sensitive.
Can a practicing surgeon detect early lymphedema reliably?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """While the study's data suggests that a surgeon could potentially detect early lymphedema using circumference measurements, it doesn't guarantee the reliability of these methods as they can also result in missed cases or false positives. The surgeon's detection ability could vary based on the method chosen and their training in these techniques. Hence, without comprehensive data, including both sensitivity and specificity of the detection methods, we can't definitively conclude whether a practicing surgeon can reliably detect early lymphedema.

## Answer
B""",
        },
        {
            "question": """Context: This study aims to evaluate local failure patterns in node negative breast cancer patients treated with post-mastectomy radiotherapy including internal mammary chain only.
Context: Retrospective analysis of 92 internal or central-breast node-negative tumours with mastectomy and external irradiation of the internal mammary chain at the dose of 50 Gy, from 1994 to 1998.
Context: Local recurrence rate was 5 % (five cases). Recurrence sites were the operative scare and chest wall. Factors associated with increased risk of local failure were age<or = 40 years and tumour size greater than 20mm, without statistical significance.
Should chest wall irradiation be included after mastectomy and negative node breast cancer?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """The context provides some information on local failure rates following mastectomy and irradiation of the internal mammary chain in node-negative breast cancer patients. It mentions that the local recurrence sites were the operative scar and chest wall. However, it does not provide information on the outcomes of patients who underwent chest wall irradiation following mastectomy, or compare outcomes between patients who received this additional treatment and those who did not. Therefore, it is not possible to make an evidence-based recommendation on whether chest wall irradiation should be included after mastectomy in these patients based on the provided context.

## Answer
B""",
        },
        {
            "question": """Context: Sulfasalazine is a widely used anti-inflammatory agent in the treatment of inflammatory bowel disease and several rheumatological disorders. Although as many as 20% of treated patients may experience reversible, dose-dependent side effects, less frequent but potentially severe, systemic reactions have also been reported.
Context: A severe systemic reaction to sulfasalazine developed in a 21-year old female with rheumatoid arthritis characterized by eosinophilia, granulomatous enteritis and myelotoxicity, cholestatic hepatitis, and seizures. The clinical course and management of this patient are presented as well as a review of the incidence and outcome of severe systemic reactions to sulfasalazine.
Is it Crohn's disease?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """The context mentions that the patient developed a severe systemic reaction to sulfasalazine, which was being used to treat rheumatoid arthritis. The side effects include eosinophilia, granulomatous enteritis and myelotoxicity, cholestatic hepatitis, and seizures. These are potential side effects of the drug, but are not specific symptoms of Crohn's disease.

## Answer
B""",
        },
        {
            "question": """Context: Tacrolimus is a potent immunosuppressive drug used in organ transplantation. Because of its substantial toxic effects, narrow therapeutic index, and interindividual pharmacokinetic variability, therapeutic drug monitoring of whole-blood tacrolimus concentrations has been recommended. We investigated the comparability of the results of 2 immunoassay systems, affinity column-mediated immunoassay (ACMIA) and microparticle enzyme immunoassay (MEIA), comparing differences in the tacrolimus concentrations measured by the 2 methods in relation to the hematologic and biochemical values of hepatic and renal functions.
Context: A total of 154 samples from kidney or liver transplant recipients were subjected to Dimension RxL HM with a tacrolimus Flex reagent cartilage for the ACMIA method and IMx tacrolimus II for the MEIA method.
Context: Tacrolimus concentrations measured by the ACMIA method (n = 154) closely correlated with those measured by the MEIA method (r = 0.84). The Bland-Altman plot using concentration differences between the 2 methods and the average of the 2 methods showed no specific trends. The tacrolimus levels determined by both the MEIA method and the ACMIA method were not influenced by hematocrit levels, but the difference between the 2 methods (ACMIA - MEIA) tended to be larger in low hematocrit samples (P<.001).
Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """The context states that the tacrolimus concentrations measured by the affinity column-mediated immunoassay (ACMIA) method closely correlated with those measured by the microparticle enzyme immunoassay (MEIA) method (r = 0.84). Additionally, the Bland-Altman plot did not show any specific trends between the two methods. This suggests that the ACMIA method may be suitable as an alternative to the MEIA method for blood tacrolimus assays. However, one notable difference is that the discrepancy between the two methods tends to be larger in low hematocrit samples.

## Answer
A""",
        },
        {
            "question": """Context: Tacrolimus is a potent immunosuppressive drug used in organ transplantation. Because of its substantial toxic effects, narrow therapeutic index, and interindividual pharmacokinetic variability, therapeutic drug monitoring of whole-blood tacrolimus concentrations has been recommended. We investigated the comparability of the results of 2 immunoassay systems, affinity column-mediated immunoassay (ACMIA) and microparticle enzyme immunoassay (MEIA), comparing differences in the tacrolimus concentrations measured by the 2 methods in relation to the hematologic and biochemical values of hepatic and renal functions.
Context: A total of 154 samples from kidney or liver transplant recipients were subjected to Dimension RxL HM with a tacrolimus Flex reagent cartilage for the ACMIA method and IMx tacrolimus II for the MEIA method.
Context: Tacrolimus concentrations measured by the ACMIA method (n = 154) closely correlated with those measured by the MEIA method (r = 0.84). The Bland-Altman plot using concentration differences between the 2 methods and the average of the 2 methods showed no specific trends. The tacrolimus levels determined by both the MEIA method and the ACMIA method were not influenced by hematocrit levels, but the difference between the 2 methods (ACMIA - MEIA) tended to be larger in low hematocrit samples (P<.001).
Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """The context states that the tacrolimus concentrations measured by the affinity column-mediated immunoassay (ACMIA) method closely correlated with those measured by the microparticle enzyme immunoassay (MEIA) method (r = 0.84). Additionally, the Bland-Altman plot did not show any specific trends between the two methods. This suggests that the ACMIA method may be suitable as an alternative to the MEIA method for blood tacrolimus assays. However, one notable difference is that the discrepancy between the two methods tends to be larger in low hematocrit samples.

## Answer
A""",
        },
        {
            "question": """Context: Treatment of obstructive hydrocephalus in children with tuberculous meningitis (TBM) depends on the level of the cerebrospinal fluid (CSF) block. Air-encephalography is regarded as the gold standard for differentiating communicating and non-communicating hydrocephalus. Since air-encephalography involves a lumbar puncture, it carries the risk of cerebral herniation. AIM. The aim of this study was to determine whether communicating and non-communicating hydrocephalus in TBM can be differentiated by means of cranial computerised tomography (CT).
Context: A number of CT indices were measured in 50 children with communicating and 34 children with non-communicating hydrocephalus according to air-encephalographic findings.
Context: The only CT finding that correlated with the type of hydrocephalus was the shape of the third ventricle. Significantly more children with non-communicating hydrocephalus had a rounded third ventricle than those with communicating hydrocephalus.
Can CT predict the level of CSF block in tuberculous hydrocephalus?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """The context mentioned that the shape of the third ventricle on a CT scan correlated with the type of hydrocephalus. Specifically, a rounded third ventricle was found significantly more often in children with non-communicating hydrocephalus than in those with communicating hydrocephalus. Therefore, this suggests that CT scans could potentially be used to predict the level of CSF block in tuberculous hydrocephalus.

## Answer
A""",
        },
        {
            "question": """Context: The study was carried on 2096 school children (1043 male, 1053 female) in Ankara. Their mean age was 9.03 years. Demographic properties of the study group and their families were determined and the serum lipid levels of the subjects were obtained. The relation between these demographic properties and lipid levels were investigated.
Context: In 135 of the subjects' serum cholesterol level was>or=200 mg/dL and in 83 subjects serum LDL-cholesterol level was>or=130 mg/dL. Despite 64.4% of the subjects reported a family history of hyperlipidemia, no relations between family history and serum lipid levels were found.
Cholesterol screening in school children: is family history reliable to choose the ones to screen?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """The context states that despite 64.4% of the subjects reporting a family history of hyperlipidemia, there was no observed relationship between family history and serum lipid levels. This suggests that family history may not be a reliable factor in determining which school children should be screened for cholesterol.

## Answer
A""",
        },
        {
            "question": """Context: Using polymerase chain reaction techniques, we evaluated the presence of HPV infection in human breast milk collected from 21 HPV-positive and 11 HPV-negative mothers.
Context: Of the 32 studied human milk specimens, no 'high-risk' HPV 16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58 or 58 DNA was detected.
Can 'high-risk' human papillomaviruses (HPVs) be detected in human breast milk?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """The context states that none of the studied human milk specimens (from both HPV-positive and HPV-negative mothers) were found to contain any 'high-risk' HPV types. Therefore, it provides enough information to suggest that 'high-risk' HPVs were not detected in human breast milk in this particular study.

## Answer
A""",
        },
    ],
}

logprobs_5shots_enough_or_not = {
    "prompt_name": "logprobs_5shots_enough_or_not",
    "response_type": "logprobs",
    "prompt": Template(
        """{% for item in examples %}## Question
{{ item.question }}
## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
    ),
    "examples": [
        {
            "question": """Context: Lymphedema may be identified by simpler circumference changes as compared with changes in limb volume.
Context: Ninety breast cancer patients were prospectively enrolled in an academic trial, and seven upper extremity circumferences were measured quarterly for 3 years. A 10% volume increase or greater than 1 cm increase in arm circumference identified lymphedema with verification by a lymphedema specialist. Sensitivity and specificity of several different criteria for detecting lymphedema were compared using the academic trial as the standard.
Context: Thirty-nine cases of lymphedema were identified by the academic trial. Using a 10% increase in circumference at two sites as the criterion, half the lymphedema cases were detected (sensitivity 37%). When using a 10% increase in circumference at any site, 74.4% of cases were detected (sensitivity 49%). Detection by a 5% increase in circumference at any site was 91% sensitive.
Can a practicing surgeon detect early lymphedema reliably?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """B""",
        },
        {
            "question": """Context: This study aims to evaluate local failure patterns in node negative breast cancer patients treated with post-mastectomy radiotherapy including internal mammary chain only.
Context: Retrospective analysis of 92 internal or central-breast node-negative tumours with mastectomy and external irradiation of the internal mammary chain at the dose of 50 Gy, from 1994 to 1998.
Context: Local recurrence rate was 5 % (five cases). Recurrence sites were the operative scare and chest wall. Factors associated with increased risk of local failure were age<or = 40 years and tumour size greater than 20mm, without statistical significance.
Should chest wall irradiation be included after mastectomy and negative node breast cancer?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """B""",
        },
        {
            "question": """Context: Sulfasalazine is a widely used anti-inflammatory agent in the treatment of inflammatory bowel disease and several rheumatological disorders. Although as many as 20% of treated patients may experience reversible, dose-dependent side effects, less frequent but potentially severe, systemic reactions have also been reported.
Context: A severe systemic reaction to sulfasalazine developed in a 21-year old female with rheumatoid arthritis characterized by eosinophilia, granulomatous enteritis and myelotoxicity, cholestatic hepatitis, and seizures. The clinical course and management of this patient are presented as well as a review of the incidence and outcome of severe systemic reactions to sulfasalazine.
Is it Crohn's disease?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """B""",
        },
        {
            "question": """Context: Tacrolimus is a potent immunosuppressive drug used in organ transplantation. Because of its substantial toxic effects, narrow therapeutic index, and interindividual pharmacokinetic variability, therapeutic drug monitoring of whole-blood tacrolimus concentrations has been recommended. We investigated the comparability of the results of 2 immunoassay systems, affinity column-mediated immunoassay (ACMIA) and microparticle enzyme immunoassay (MEIA), comparing differences in the tacrolimus concentrations measured by the 2 methods in relation to the hematologic and biochemical values of hepatic and renal functions.
Context: A total of 154 samples from kidney or liver transplant recipients were subjected to Dimension RxL HM with a tacrolimus Flex reagent cartilage for the ACMIA method and IMx tacrolimus II for the MEIA method.
Context: Tacrolimus concentrations measured by the ACMIA method (n = 154) closely correlated with those measured by the MEIA method (r = 0.84). The Bland-Altman plot using concentration differences between the 2 methods and the average of the 2 methods showed no specific trends. The tacrolimus levels determined by both the MEIA method and the ACMIA method were not influenced by hematocrit levels, but the difference between the 2 methods (ACMIA - MEIA) tended to be larger in low hematocrit samples (P<.001).
Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """A""",
        },
        {
            "question": """Context: Tacrolimus is a potent immunosuppressive drug used in organ transplantation. Because of its substantial toxic effects, narrow therapeutic index, and interindividual pharmacokinetic variability, therapeutic drug monitoring of whole-blood tacrolimus concentrations has been recommended. We investigated the comparability of the results of 2 immunoassay systems, affinity column-mediated immunoassay (ACMIA) and microparticle enzyme immunoassay (MEIA), comparing differences in the tacrolimus concentrations measured by the 2 methods in relation to the hematologic and biochemical values of hepatic and renal functions.
Context: A total of 154 samples from kidney or liver transplant recipients were subjected to Dimension RxL HM with a tacrolimus Flex reagent cartilage for the ACMIA method and IMx tacrolimus II for the MEIA method.
Context: Tacrolimus concentrations measured by the ACMIA method (n = 154) closely correlated with those measured by the MEIA method (r = 0.84). The Bland-Altman plot using concentration differences between the 2 methods and the average of the 2 methods showed no specific trends. The tacrolimus levels determined by both the MEIA method and the ACMIA method were not influenced by hematocrit levels, but the difference between the 2 methods (ACMIA - MEIA) tended to be larger in low hematocrit samples (P<.001).
Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """A""",
        },
        {
            "question": """Context: Treatment of obstructive hydrocephalus in children with tuberculous meningitis (TBM) depends on the level of the cerebrospinal fluid (CSF) block. Air-encephalography is regarded as the gold standard for differentiating communicating and non-communicating hydrocephalus. Since air-encephalography involves a lumbar puncture, it carries the risk of cerebral herniation. AIM. The aim of this study was to determine whether communicating and non-communicating hydrocephalus in TBM can be differentiated by means of cranial computerised tomography (CT).
Context: A number of CT indices were measured in 50 children with communicating and 34 children with non-communicating hydrocephalus according to air-encephalographic findings.
Context: The only CT finding that correlated with the type of hydrocephalus was the shape of the third ventricle. Significantly more children with non-communicating hydrocephalus had a rounded third ventricle than those with communicating hydrocephalus.
Can CT predict the level of CSF block in tuberculous hydrocephalus?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """A""",
        },
        {
            "question": """Context: The study was carried on 2096 school children (1043 male, 1053 female) in Ankara. Their mean age was 9.03 years. Demographic properties of the study group and their families were determined and the serum lipid levels of the subjects were obtained. The relation between these demographic properties and lipid levels were investigated.
Context: In 135 of the subjects' serum cholesterol level was>or=200 mg/dL and in 83 subjects serum LDL-cholesterol level was>or=130 mg/dL. Despite 64.4% of the subjects reported a family history of hyperlipidemia, no relations between family history and serum lipid levels were found.
Cholesterol screening in school children: is family history reliable to choose the ones to screen?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """A""",
        },
        {
            "question": """Context: Using polymerase chain reaction techniques, we evaluated the presence of HPV infection in human breast milk collected from 21 HPV-positive and 11 HPV-negative mothers.
Context: Of the 32 studied human milk specimens, no 'high-risk' HPV 16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58 or 58 DNA was detected.
Can 'high-risk' human papillomaviruses (HPVs) be detected in human breast milk?

A. The context gives enough information to answer the question.
B. The context does not give enough information to answer the question.
""",
            "answer": """A""",
        },
    ],
}

logprobs_5shots_yes_or_not = {
    "prompt_name": "logprobs_5shots_enough_or_not",
    "response_type": "logprobs",
    "prompt": Template(
        """{% for item in examples %}## Question
{{ item.question }}
## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
    ),
    "examples": [
        {
            "question": """Context: Tacrolimus is a potent immunosuppressive drug used in organ transplantation. Because of its substantial toxic effects, narrow therapeutic index, and interindividual pharmacokinetic variability, therapeutic drug monitoring of whole-blood tacrolimus concentrations has been recommended. We investigated the comparability of the results of 2 immunoassay systems, affinity column-mediated immunoassay (ACMIA) and microparticle enzyme immunoassay (MEIA), comparing differences in the tacrolimus concentrations measured by the 2 methods in relation to the hematologic and biochemical values of hepatic and renal functions.
Context: A total of 154 samples from kidney or liver transplant recipients were subjected to Dimension RxL HM with a tacrolimus Flex reagent cartilage for the ACMIA method and IMx tacrolimus II for the MEIA method.
Context: Tacrolimus concentrations measured by the ACMIA method (n = 154) closely correlated with those measured by the MEIA method (r = 0.84). The Bland-Altman plot using concentration differences between the 2 methods and the average of the 2 methods showed no specific trends. The tacrolimus levels determined by both the MEIA method and the ACMIA method were not influenced by hematocrit levels, but the difference between the 2 methods (ACMIA - MEIA) tended to be larger in low hematocrit samples (P<.001).
Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay?

A. yes
B. no
""",
            "answer": """A""",
        },
        {
            "question": """Context: Treatment of obstructive hydrocephalus in children with tuberculous meningitis (TBM) depends on the level of the cerebrospinal fluid (CSF) block. Air-encephalography is regarded as the gold standard for differentiating communicating and non-communicating hydrocephalus. Since air-encephalography involves a lumbar puncture, it carries the risk of cerebral herniation. AIM. The aim of this study was to determine whether communicating and non-communicating hydrocephalus in TBM can be differentiated by means of cranial computerised tomography (CT).
Context: A number of CT indices were measured in 50 children with communicating and 34 children with non-communicating hydrocephalus according to air-encephalographic findings.
Context: The only CT finding that correlated with the type of hydrocephalus was the shape of the third ventricle. Significantly more children with non-communicating hydrocephalus had a rounded third ventricle than those with communicating hydrocephalus.
Can CT predict the level of CSF block in tuberculous hydrocephalus?

A. yes
B. no
""",
            "answer": """A""",
        },
        {
            "question": """Context: The study was carried on 2096 school children (1043 male, 1053 female) in Ankara. Their mean age was 9.03 years. Demographic properties of the study group and their families were determined and the serum lipid levels of the subjects were obtained. The relation between these demographic properties and lipid levels were investigated.
Context: In 135 of the subjects' serum cholesterol level was>or=200 mg/dL and in 83 subjects serum LDL-cholesterol level was>or=130 mg/dL. Despite 64.4% of the subjects reported a family history of hyperlipidemia, no relations between family history and serum lipid levels were found.
Cholesterol screening in school children: is family history reliable to choose the ones to screen?

A. yes
B. no
""",
            "answer": """B""",
        },
        {
            "question": """Context: Using polymerase chain reaction techniques, we evaluated the presence of HPV infection in human breast milk collected from 21 HPV-positive and 11 HPV-negative mothers.
Context: Of the 32 studied human milk specimens, no 'high-risk' HPV 16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58 or 58 DNA was detected.
Can 'high-risk' human papillomaviruses (HPVs) be detected in human breast milk?

A. yes
B. no
""",
            "answer": """B""",
        },
    ],
}


cot_yes_or_not = {
    "prompt_name": "cot_yes_or_not",
    "response_type": "answer_letter",
    "prompt": Template(
        """{% for item in examples %}## Question
{{ item.question }}
## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
    ),
    "examples": [
        {
            "question": """Context: Tacrolimus is a potent immunosuppressive drug used in organ transplantation. Because of its substantial toxic effects, narrow therapeutic index, and interindividual pharmacokinetic variability, therapeutic drug monitoring of whole-blood tacrolimus concentrations has been recommended. We investigated the comparability of the results of 2 immunoassay systems, affinity column-mediated immunoassay (ACMIA) and microparticle enzyme immunoassay (MEIA), comparing differences in the tacrolimus concentrations measured by the 2 methods in relation to the hematologic and biochemical values of hepatic and renal functions.
Context: A total of 154 samples from kidney or liver transplant recipients were subjected to Dimension RxL HM with a tacrolimus Flex reagent cartilage for the ACMIA method and IMx tacrolimus II for the MEIA method.
Context: Tacrolimus concentrations measured by the ACMIA method (n = 154) closely correlated with those measured by the MEIA method (r = 0.84). The Bland-Altman plot using concentration differences between the 2 methods and the average of the 2 methods showed no specific trends. The tacrolimus levels determined by both the MEIA method and the ACMIA method were not influenced by hematocrit levels, but the difference between the 2 methods (ACMIA - MEIA) tended to be larger in low hematocrit samples (P<.001).
Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay?

A. yes
B. no
""",
            "answer": """Based on the provided study results, it would appear that the Affinity Column-Mediated Immunoassay (ACMIA) method could serve as an alternative to the Microparticle Enzyme Immunoassay (MEIA) method for blood tacrolimus assays. The high correlation (r = 0.84) between the results from the two methods indicates a strong relationship. Furthermore, the Bland-Altman plot showed no specific trends, implying that the differences between the two methods were not systematic and could be random. While the study did note a larger difference between the two methods in samples with low hematocrit levels, this does not necessarily mean that one method is less accurate than the other. However, before concluding, it would be crucial to conduct additional studies to confirm these findings and evaluate potential impacts on clinical decision-making. For instance, research should investigate whether the larger difference at low hematocrit levels could affect patient outcomes. Hence, the most accurate answer at this moment would be

Answer: A""",
        },
        {
            "question": """Context: Treatment of obstructive hydrocephalus in children with tuberculous meningitis (TBM) depends on the level of the cerebrospinal fluid (CSF) block. Air-encephalography is regarded as the gold standard for differentiating communicating and non-communicating hydrocephalus. Since air-encephalography involves a lumbar puncture, it carries the risk of cerebral herniation. AIM. The aim of this study was to determine whether communicating and non-communicating hydrocephalus in TBM can be differentiated by means of cranial computerised tomography (CT).
Context: A number of CT indices were measured in 50 children with communicating and 34 children with non-communicating hydrocephalus according to air-encephalographic findings.
Context: The only CT finding that correlated with the type of hydrocephalus was the shape of the third ventricle. Significantly more children with non-communicating hydrocephalus had a rounded third ventricle than those with communicating hydrocephalus.
Can CT predict the level of CSF block in tuberculous hydrocephalus?

A. yes
B. no
""",
            "answer": """Based on the information provided, it seems that cranial computed tomography (CT) could help in differentiating the type of hydrocephalus in patients with tuberculous meningitis (TBM). The shape of the third ventricle was found to be significantly different between patients with communicating and non-communicating hydrocephalus. Therefore, a CT could potentially be used to predict the level of cerebrospinal fluid (CSF) block, thereby indicating the type of hydrocephalus. However, it's worth noting that the data presented only mentions one CT finding that correlated with the type of hydrocephalus. To ensure the accuracy of these results and to explore the full potential of CT as a diagnostic tool in this context, further studies with larger sample sizes and more varied parameters are needed.

Answer: A""",
        },
        {
            "question": """Context: The study was carried on 2096 school children (1043 male, 1053 female) in Ankara. Their mean age was 9.03 years. Demographic properties of the study group and their families were determined and the serum lipid levels of the subjects were obtained. The relation between these demographic properties and lipid levels were investigated.
Context: In 135 of the subjects' serum cholesterol level was>or=200 mg/dL and in 83 subjects serum LDL-cholesterol level was>or=130 mg/dL. Despite 64.4% of the subjects reported a family history of hyperlipidemia, no relations between family history and serum lipid levels were found.
Cholesterol screening in school children: is family history reliable to choose the ones to screen?

A. yes
B. no
""",
            "answer": """
Based on the data provided in the study, family history did not show any correlation with serum lipid levels among the school children studied. Despite a significant proportion (64.4%) of the subjects reporting a family history of hyperlipidemia, no relationship was found between this family history and the actual serum lipid levels of the subjects. Therefore, relying on family history alone may not be a reliable method for choosing which children to screen for high cholesterol. Further criteria may need to be considered for a more accurate selection process.

Answer: B""",
        },
        {
            "question": """Context: Using polymerase chain reaction techniques, we evaluated the presence of HPV infection in human breast milk collected from 21 HPV-positive and 11 HPV-negative mothers.
Context: Of the 32 studied human milk specimens, no 'high-risk' HPV 16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58 or 58 DNA was detected.
Can 'high-risk' human papillomaviruses (HPVs) be detected in human breast milk?

A. yes
B. no
""",
            "answer": """According to the study provided, none of the 'high-risk' HPV types, including HPV 16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58, or 58, were detected in any of the 32 human milk samples collected from the 21 HPV-positive and 11 HPV-negative mothers. Therefore, based on the results of this specific study, 'high-risk' HPVs cannot be detected in human breast milk.

Answer: B""",
        },
    ],
}
