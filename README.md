
# 🧠 Response Time Estimation Using Physiological Signals  

---

## Step 1. Data Analysis  

### 1.1 Dataset and Research Paper  
- **Paper Link:** [A Multisensor Dataset of South Asian Post-Graduate Students Working on Mental Rotation Tasks](https://www.nature.com/articles/s41597-025-04865-5#Sec19)  
- **Dataset Link:** [SpringerNature Dataset](https://springernature.figshare.com/articles/dataset/A_multisensor_dataset_of_south_asian_post-graduate_students_working_on_mental_rotation_tasks/28120670?file=51439640)  

#### Summary of Dataset (38 participants)  
- Participants performed **mental rotation tasks** while multiple physiological and behavioral signals were recorded.  
- Data captures **how people feel, react, and perform** on cognitive tasks.  

**Signals Collected:**  
1. **Galvanic Skin Response (GSR):** Measures palm sweating → emotional stress.  
2. **Electroencephalography (EEG):** Brain activity using electrodes.  
3. **Screen Recording:** Activity on participant’s screen.  
4. **Facial Expressions:** Emotions from face detection.  
5. **Manual Emotion Logging:** Participants’ self-reported feelings before/after tasks.  
6. **Task Performance Logs:** Response time, correctness, and question difficulty.  
7. **Eye-tracking (ET):** Gaze location, pupil size, fixation patterns.  
8. **Self-Reports:** Strategies and emotional state.  
9. **Mental Rotation Tasks:** Puzzle-like tasks requiring mental rotation of 3D objects.  

**Experimental Conditions:**  
- Category 1: No time restriction, no feedback  
- Category 2: No time restriction, with feedback  
- Category 3: Time restriction, no feedback  

**Research Insights:**  
- **Gender differences** in rotation tasks disappear when time limits are removed.  
- **Strategies observed:**  
  - *Holistic rotation* (rare but efficient).  
  - *Piecemeal rotation* (common, rotating parts).  
- **Eye-tracking:** Fixation and saccade patterns show strategy.  
- **EEG:** Provides insights into approach/avoidance and overload.  
- **GSR & Facial expressions:** Reveal affective state.  

---

### 1.2 EEG Data Analysis  

We analyzed EEG brain waves per question to check for insights:  
1. Overall wave patterns during problem-solving were irregular.  
2. At the **start**, brain waves rise.  
3. In the **middle**, they start to fall.  
4. At the **end**, they drop further.  
5. **TP9 channel:** Fluctuations exist but remain mostly stable.  

**Correlation Analysis (from covariance matrix):**  
- Strong positive correlations:  
  - Beta_AF7 & Beta_AF8 (0.63)  
  - Beta_AF8 & Gamma_TP9 (0.69)  
  - Gamma_TP9 & Gamma_AF7 (0.89)  
  - Beta_TP10 & Gamma_TP10 (0.92)  
- Weak correlations between **Category** and brain waves → task type does not strongly affect EEG.  

---

## Step 2. Final Problem Statement  

**Problem:** *Response Time Estimation Using Physiological Signals*  

---

## Step 3. Dataset Construction  

To predict response time, we selected important features and created a subset:  

1. **PSY.csv:** `QuestionKey`, `Category`, `Response Time`, + `Participant ID`.  
2. **EEG.csv:** Delta, Theta, Alpha, Beta at (TP9, AF7, AF8, TP10).  
   - Extracted: `mean`, `median`, `std`, `min`, `max` per wave.  
3. **EYE.csv:** Gaze (x, y), Pupil, Distance.  
   - Extracted: `mean`, `median`, `std`, `min`, `max`.  
4. **IVT.csv:** Fixation, Saccade, Gaze velocity/acceleration.  
   - Derived features:  
     - Avg Fixation Duration  
     - Avg Saccade Amplitude  
     - Total Scanpath Length (sum of saccades)  
     - Mean Gaze Velocity  
     - Mean Gaze Acceleration  
     - Std Gaze Velocity  

✅ All merged using **Participant ID**.  
✅ Across 38 participants → **132 features** (128 numerical + 4 categorical).  
✅ Final subset: `EEG_IVT_EYE_final_merged_data.csv`.  

---

## Step 4. Model Training – Attempt 1  

- Only **averages** of EEG & eye features were used.  
- Target: `Response Time`.  
- Features: EEG + Eye averages.  
- Combined across all 38 participants (QuestionKey 1–38).  

**Results (poor performance):**  

| Model              | R²    | MAE   | RMSE   |
|--------------------|-------|-------|--------|
| Linear Regression  | 0.117 | 9.19  | 13.12  |
| Ridge Regression   | 0.118 | 9.17  | 13.11  |
| Lasso Regression   | 0.117 | 9.19  | 13.12  |
| Random Forest      | 0.243 | 8.51  | 12.16  |
| Gradient Boosting  | 0.318 | 7.88  | 11.53  |
| XGBoost            | 0.303 | 8.05  | 11.66  |

🔴 **Only ~30% variance explained → too low.**  

---

## Step 5. Model Training – Final Subset  

To improve, we used **min, max, mean, median, std** features instead of only averages.  
This gave richer feature representation.  

**Results (final performance):**  

| Model                  | R²     | RMSE   |
|-------------------------|--------|--------|
| XGBoost                | 0.9374 | 3.3564 |
| Gradient Boosting       | 0.9336 | 3.4558 |
| Random Forest           | 0.9200 | 3.7925 |
| Ridge Regression        | 0.8981 | 4.2809 |
| Support Vector Machine  | 0.4210 | 10.208 |

✅ **Best Model:** XGBoost Regressor  

---

### XGBoost Evaluation  

- **R²:** 0.9374 → Explains ~93.7% of variance.  
- **RMSE:** 3.3564 → Avg error ≈ 3.36 units.  
- **Interpretation:**  
  - Positive error → model underestimated.  
  - Negative error → model overestimated.  
  - Predictions are very close to actual.  

**Sample Predictions:**  

| Actual | Predicted | Error |
|--------|-----------|-------|
| 14.73  | 14.94     | -0.20 |
| 12.46  | 11.53     | 0.92  |
| 26.58  | 23.35     | 3.23  |
| 6.47   | 5.63      | 0.84  |
| 18.05  | 19.45     | -1.40 |

**Feature Importance:**  
- Key EEG & Eye features were identified as most influential.  
- Rich feature set allowed the model to capture trends effectively.  

---

## ✅ Conclusion  

- **Cleaning, scaling, and feature extraction** were crucial.  
- Only using averages → poor results.  
- Using **min, max, median, mean, std** + IVT features → excellent results.  
- **XGBoost** achieved the **best performance** with R² ≈ 0.94.  
- Project proves that **response time can be reliably estimated from physiological signals**.  

---
