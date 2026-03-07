# Pad-Line
Pad-Line is a lightweight clinical timeline visualization tool for oncology pharmacotherapy management.  
It integrates **adverse events (AEs)** and **clinical actions** (e.g., hold / dose reduction) on a single patient-centric time axis to reduce cognitive load during case review and conferences.

**Japanese README:** `README_JA.md`

> **Scope:** Visualization & workflow support  
> **Not** a clinical decision recommendation system.

---

## Background
In daily clinical practice, relevant information is often fragmented across free-text notes and isolated time points.  
This makes it time-consuming to reconstruct:
- when an AE started and how long it persisted,
- when an action (hold / dose reduction / restart) occurred,
- how multiple events overlapped over time.

Pad-Line focuses on **N=1 case understanding** (safe management and efficient discussion).  
If structured data accumulates naturally, the same format can be reused for **multi-patient comparison** (swimmer plot) and further analysis.

---

## Key Features
- **Swimmer / Swimlane toggle**
  - *Swimmer:* overview across multiple patients
  - *Swimlane:* single-patient view with category-based lanes
- **AE duration visualization**
  - start marker + duration line (color-coded by Grade)
- **Action events**
  - point events for actions such as hold / dose reduction
- **Ongoing event rendering**
  - ongoing AEs are shown with an arrow (→) when end date is missing
- **Automatic overlap avoidance**
  - same-day events are jittered to remain readable
- **Clinician-friendly controls**
  - a single slider adjusts internal layout parameters for readability

---

## Screenshots
All screenshots are generated from **synthetic data** (no real patient data).

### Swimlane (single patient)
![Swimlane](images/swimlane.png)

### Swimmer (multi-patient overview)
![Swimmer](images/swimmer.png)
---

## Quickstart
### 1) Install
```bash
pip install -r requirements.txt
streamlit run app.py
