# 🌬️ Climate-Aware Respiratory Risk Forecasting System (CARRFS)

The **Climate-Aware Respiratory Risk Forecasting System (CARRFS)** is an AI-powered platform that predicts the Air Quality Index (AQI) using machine learning and real-time environmental data. It helps users understand air pollution levels, assess respiratory health risks, and make safer travel decisions. 

Whether you're an asthma patient planning your day, a healthcare provider monitoring local conditions, or a fitness enthusiast seeking the cleanest jogging routes, CARRFS acts as your personalized health co-pilot.

Built using **Flask**, **XGBoost**, the **Open-Meteo API**, and the **Groq API (LLM)**, this tool provides deep insights into:

- Real-time respiratory risk forecasting
- Personalized medical precautions
- Clean route optimization
- Historical AQI trend analysis

---

## 🧠 Project Overview

**“Your AI-Powered Respiratory Health Co-Pilot.”**

CARRFS combines real-time environmental intelligence with Machine Learning and LLM-based summarization to transform complex climate data into personalized, structured health advice.

---

## ⚙️ Tools & Tech Stack

| Component | Tool / API |
| :--- | :--- |
| **Weather/AQI API** | Open-Meteo API |
| **Language Model** | Groq API (LLM Assistant) |
| **Machine Learning** | XGBoost & Scikit-learn |
| **Backend** | Python (Flask) |
| **Database** | SQLite |
| **Frontend** | HTML, CSS, JavaScript |

---

## 📦 Project Architecture

```text
User Input (Health Profile & Location)
       ↓
Backend Logic (app.py)
       ↓
Open-Meteo API → Real-time Weather & AQI Data
       ↓
XGBoost Model → Risk Level Prediction
       ↓
Groq LLM → Personalized Health Advice Generation
       ↓
Dashboard UI (Actionable Insights)
```

---

## 🚀 Features

**1. 🩺 Personalized Risk Forecasting**
*   **Input:** User Health Profile (Age, Smoker status, Conditions like Asthma/COPD)
*   **Output:** Low, Moderate, or High risk prediction for outdoor activities.

**2. 🤖 AI Health Assistant**
*   **Input:** Current Air Quality & User Profile
*   **Output:** Real-time, highly tailored medical precautions and daily advice.

**3. 🗺️ Clean Route Optimizer**
*   **Input:** Source and Destination points
*   **Output:** Safest navigational route minimizing PM2.5 and pollutant exposure.

**4. 📈 Historical AQI Trends**
*   **Input:** Location data
*   **Output:** Visualized 30-day historical air quality metrics.

---

## 🧩 UI Flow

**1. Health Dashboard Flow**
```text
User Registers / Updates Health Profile
       ↓
System Fetches Live Environmental Data
       ↓
ML Models Predict Respiratory Risk
       ↓
LLM Generates Personalized Precautions
       ↓
User Views Health Dashboard
```

**2. Route Optimizer Flow**
```text
User Enters Source & Destination
       ↓
System Analyzes Multiple Navigational Routes
       ↓
Calculates Pollutant Exposure per Route
       ↓
User Views Map with Cleanest Route Highlighted
```

---

## 📌 Use Cases

| Use Case | Who Benefits | Value Provided |
| :--- | :--- | :--- |
| **Daily Health Planning** | Asthma/COPD Patients | Prevents attacks by avoiding severe pollution. |
| **Fitness & Exercise** | Runners, Cyclists | Identifies the cleanest time and route for outdoor workouts. |
| **Medical Advisory** | Healthcare Providers | Monitors local environmental health risks for patients. |

---

## 📊 Example Output

**🧪 Risk Forecast – "Asthma Patient in Gurugram"**
*   **Current AQI:** 285 (Very Unhealthy)
*   **Predicted Risk:** HIGH
*   **AI Advice:** "Limit outdoor exertion today. Ensure your inhaler is accessible. Consider exercising indoors with an air purifier."

**🗺️ Route Optimization – "Cyber Hub to Sector 56"**
*   **Route A (Golf Course Rd):** 25 mins (Avg PM2.5: 185 µg/m³)
*   **Route B (Internal Sectors):** 28 mins (Avg PM2.5: 95 µg/m³)
*   **System Recommendation:** Route B (Safer respiratory choice)

---

## ⚠️ Important Setup Instructions (API Keys Required)

To run this project on your own machine, you **MUST** provide your own Groq API key for the AI Health Assistant to function.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Keertana04/Climate-Aware-Respiratory-Risk-Forecasting-System.git
   cd Climate-Aware-Respiratory-Risk-Forecasting-System
   ```

2. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API Key:**
   * Create a new file in the root folder named exactly `.env`
   * Go to the [Groq Console](https://console.groq.com/) and generate a free API key.
   * Add the following line to your `.env` file:
     ```text
     GROQ_API_KEY=your_actual_api_key_here
     ```
   *(Note: The `.env` and `users_db` files are intentionally ignored by Git to protect your private keys and user data. You must create `.env` manually, but the database will be created automatically when the app runs!)*

4. **Run the Application:**
   ```bash
   flask run
   ```
   Or simply run `python app.py` and open your browser to `http://localhost:5000`.

---

## ⚠️ Data Scope & Limitations

*   **Geographical Constraint:** The Machine Learning models and historical trend analyses in this current version are trained specifically using an extensive dataset from **Gurugram, India**.
*   **Impact:** While the real-time API fetches accurate data for any global location, the *predictive* modeling risk scores are optimized for Gurugram's specific pollution and climate signatures.

---

## 🌱 Future Enhancements

*   **🌍 Geographical Expansion:** Scaling the Machine Learning models to support accurate predictions for other major metropolitan areas globally, not just Gurugram.
*   **📱 Mobile App Integration:** Developing dedicated iOS and Android applications using React Native for a seamless on-the-go experience.
*   **⌚ Wearable Device Sync:** Integration with Apple Watch and WearOS to push live respiratory risk alerts directly to users' wrists during outdoor activities.
*   **🚦 Traffic-Aware Routing:** Upgrading the Route Optimizer to simultaneously evaluate both real-time AQI *and* live traffic congestion to find the absolute fastest and cleanest paths.
*   **🏥 Clinical Integration:** Allowing users to securely sync their CARRFS health reports directly with their local pulmonologists or hospital systems.
*   **🔔 SMS & Email Alerts:** Implementing automated push notifications to warn users the night before a predicted severe pollution spike.
*   **🌐 Multi-Language Support:** Adding regional language options (e.g., Hindi, Marathi) to the AI Health Assistant to increase accessibility for a broader demographic.
