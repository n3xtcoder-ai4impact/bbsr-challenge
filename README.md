# Goal 1: Predict which pollutants are likely present in construction materials and building components

## 1) Collected and cleaned data
- From ökobaudat (OBD), tbaustoff (TBS) and pollutants data sets
- Mapped each material to its role (e.g., insulation, adhesive) using keywords
- Standardized noisy contaminant terms like “klebstoffe” and “kleberreste” into unified labels like klebstoff

## 2) Increased the sample size to train a MLM (see bootstrap.ipynb)
- Just by joining tbaustoff to pollutants datasets we were only getting 14 matches
- Applied Fuzzy Match + Manual Input to get high confidence pollutants-eol category pairs = 147 matches. 
- 1st bootstrap: Added pollutant label to all TBS materials by shared EOL category  - Made the assumption that same eol categories share the same pollutants. I decided to make this generalization only if treat the pollutants as likely candidates, not definitive (it’s a multi label classification problem) I will be treating the labels probabilistically, not deterministically.
- UUID merge: Assigned EOL category to matching OBD materials
- 2nd bootstrap:  Added pollutant label to OBD materials by shared EOL.
- Semi supervised learning: 
  - Trained a weak model on the existing labeled data (Random Forest )
  - Use it to predict labels for a larger pool of unlabeled materials
  - Retained high-confidence predictions (e.g., probability > 0.9) as pseudo-labeled data
- Following step (recommendation) - Leverage domain knowledge in Active learning: 
  - Use the trained model to identify the most uncertain predictions
  - Used simple logical rules (heuristics) to label just those
  - Some pollutants can be rule-assigned with decent confidence

## 3) Trained a multi label pollutant prediction models (see pollutant_prediction_clean.ipynb)
- Built a machine learning model (Random Forest with Multi Output Classifier) that predicts which pollutant classes (S0–S4) a material might contain
- Expanded this into a multi-label model that also predicts specific contaminants like bitumen, klebstoff, biozide, etc.
- Used material metadata, technology 	and EOL context as input features
- The output is the probability of different pollutants being present in a given material. 
- NOTE : the sample size is still not big enough. Only with a proper mapping between TBS and OBD we’ll be able to get more confident results (goal 2)

## 4) Added context-aware prediction
- Recognized that pollutants often come from how materials are combined 
- Mapped which materials are grouped together in building components by webscrapping bautelieditor Web app (see elca_scrapping.ipynb)
- Built models that consider both a material’s features and the pollutants from its neighboring materials
- Adjusted the original predictions based on the average risk in the component
  
##  5) Improved prediction quality
- Integrated text analysis (TF-IDF) from product names to catch contaminant clues in naming
- Filtered and prioritized the most frequent and learnable contaminants
- Applied thresholds to highlight likely pollutants (e.g., ≥ 30% probability)

## 6) Produced practical outputs
- Generated detailed tables showing:-
  - For each material: which pollutants are likely present
  - For each component: the collective contaminant risk
  -	Highlighted which contaminants are most likely to require attention

![image](https://github.com/user-attachments/assets/d566db1a-eb67-40cc-8945-671019355371)
