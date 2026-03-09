// ============================================================
// CARRFS - Premium Frontend Script
// ============================================================

async function predictAQI() {
    const btn = document.getElementById("predictBtn");
    const originalBtnText = btn.innerText;

    // Collect Inputs
    const pm25 = document.getElementById("pm25").value;
    const pm10 = document.getElementById("pm10").value;
    const temp = document.getElementById("temp").value;
    const humidity = document.getElementById("humidity").value;
    const co2 = document.getElementById("co2").value;

    if (!pm25 || !pm10 || !temp || !humidity || !co2) {
        alert("Please provide all environmental readings.");
        return;
    }

    // Set Loading State
    btn.disabled = true;
    btn.innerText = "Processing Data...";
    btn.style.opacity = "0.7";

    try {
        // API Call (using relative path for better portability)
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pm2_5: parseFloat(pm25),
                pm10: parseFloat(pm10),
                temp: parseFloat(temp),
                humidity: parseFloat(humidity),
                co2: parseFloat(co2),
                age: "General",
                gender: "Not specified",
                smoker: "No",
                health_conditions: []
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Show Result Sections
        const resultSection = document.getElementById("resultSection");
        const precautionSection = document.getElementById("precautionSection");

        resultSection.style.display = "flex";
        precautionSection.style.display = "block";

        // Update AQI Hero Section
        document.getElementById("aqiValue").innerText = data.aqi;
        document.getElementById("severityBadge").innerText = data.severity;

        // Dynamic Styling
        setSeverityStyle(data.severity);

        // Update Pollutant Cards
        updatePollutantDisplay("pm25Val", pm25, "µg/m³");
        updatePollutantDisplay("pm10Val", pm10, "µg/m³");
        updatePollutantDisplay("co2Val", co2, "ppm");
        updatePollutantDisplay("tempVal", temp, "°C");
        updatePollutantDisplay("humVal", humidity, "%");

        // Update Precautions
        const list = document.getElementById("precautions");
        list.innerHTML = "";

        if (data.precautions && data.precautions.length > 0) {
            data.precautions.forEach((p, index) => {
                const li = document.createElement("li");
                li.innerText = p;
                li.style.animation = `fadeIn 0.5s ease forwards ${index * 0.1}s`;
                li.style.opacity = "0";
                list.appendChild(li);
            });
        }

        // Scroll to results smoothly
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        console.error("Dashboard Error:", error);
        alert("Server Error: " + error.message);
    } finally {
        // Reset Button
        btn.disabled = false;
        btn.innerText = originalBtnText;
        btn.style.opacity = "1";
    }
}

function updatePollutantDisplay(id, value, unit) {
    const el = document.getElementById(id);
    el.innerHTML = `${value} <span style="font-size: 0.875rem; color: var(--text-secondary); font-weight: 500;">${unit}</span>`;
}

function setSeverityStyle(severity) {
    const hero = document.querySelector(".hero");
    const badge = document.getElementById("severityBadge");

    let gradient = "linear-gradient(135deg, rgba(31, 41, 55, 0.9), rgba(17, 24, 39, 0.9))";
    let badgeBg = "var(--card-bg)";

    switch (severity) {
        case "Good":
            gradient = "linear-gradient(135deg, #064e3b, #065f46)";
            badgeBg = "var(--good)";
            break;
        case "Satisfactory":
            gradient = "linear-gradient(135deg, #854d0e, #a16207)";
            badgeBg = "var(--satisfactory)";
            break;
        case "Moderately Polluted":
            gradient = "linear-gradient(135deg, #9a3412, #c2410c)";
            badgeBg = "var(--moderate)";
            break;
        case "Poor":
            gradient = "linear-gradient(135deg, #991b1b, #b91c1c)";
            badgeBg = "var(--poor)";
            break;
        case "Very Poor":
            gradient = "linear-gradient(135deg, #5b21b6, #6d28d9)";
            badgeBg = "var(--verypoor)";
            break;
        case "Severe":
            gradient = "linear-gradient(135deg, #111827, #000000)";
            badgeBg = "var(--severe)";
            break;
    }

    hero.style.background = gradient;
    badge.style.background = badgeBg;
    badge.style.color = (severity === "Satisfactory") ? "#000" : "#fff";
}
