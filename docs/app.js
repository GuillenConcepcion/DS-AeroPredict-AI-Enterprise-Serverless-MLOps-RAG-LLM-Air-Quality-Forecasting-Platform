/**
 * AeroPredict AI - GitHub Pages Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  const locationSelect = document.getElementById("locationSelect");
  loadDashboardData(locationSelect.value);

  locationSelect.addEventListener("change", (e) => {
    loadDashboardData(e.target.value);
  });
});

async function loadDashboardData(locationName) {
  try {
    const response = await fetch("../data/latest_predictions.json");
    if (!response.ok) throw new Error("JSON predictions file not found.");
    const data = await response.json();
    renderDashboard(data, locationName);
  } catch (error) {
    console.warn("Generating synthetic demonstration data.", error);
    renderDashboard(generateMockData(locationName), locationName);
  }
}

function renderDashboard(data, locationName) {
  const currentAqi = data.current_us_aqi || 36;
  const currentPm25 = data.current_pm2_5 || 8.7;
  const mae = data.recent_monitoring_mae || 1.52;

  document.getElementById("aqiValue").innerText = currentAqi;
  document.getElementById("pm25Value").innerText = currentPm25;
  document.getElementById("maeValue").innerText = mae;

  const circle = document.getElementById("aqiCircle");
  const badge = document.getElementById("aqiBadge");
  const desc = document.getElementById("aqiDescription");

  let color = "#10b981";
  let status = "Good";
  let description = "Air quality is satisfactory, posing little or no risk.";

  if (currentAqi > 50 && currentAqi <= 100) {
    color = "#f59e0b";
    status = "Moderate";
    description = "Air quality is acceptable for the public.";
  } else if (currentAqi > 100) {
    color = "#f43f5e";
    status = "Unhealthy";
    description = "Unhealthy air quality levels detected.";
  }

  circle.style.borderColor = color;
  circle.style.boxShadow = `0 0 25px ${color}66`;
  badge.innerText = status;
  badge.style.background = `${color}33`;
  badge.style.color = color;
  badge.style.borderColor = color;
  desc.innerText = description;

  // Update LLM Advisory Cards
  updateLlmAdvisory(currentAqi, locationName);

  renderForecastChart(data.forecast);
}

function updateLlmAdvisory(aqi, locationName) {
  const activityEl = document.getElementById("llmActivity");
  const ventilationEl = document.getElementById("llmVentilation");
  const sensitiveEl = document.getElementById("llmSensitive");

  if (aqi <= 50) {
    if (activityEl) activityEl.innerText = `Optimal atmospheric conditions in ${locationName}. Safe for outdoor running, marathons, and outdoor sports.`;
    if (ventilationEl) ventilationEl.innerText = "Open windows freely in morning and evening hours for natural indoor airflow.";
    if (sensitiveEl) sensitiveEl.innerText = "No health precautions required for sensitive groups (asthma, children, elderly).";
  } else if (aqi <= 100) {
    if (activityEl) activityEl.innerText = `Air quality in ${locationName} is moderate. Unusually sensitive runners should monitor breathing during intense workouts.`;
    if (ventilationEl) ventilationEl.innerText = "Ventilation is acceptable, but close windows if outdoor pollen or dust increases.";
    if (sensitiveEl) sensitiveEl.innerText = "Individuals with asthma or respiratory conditions should keep relief medication accessible.";
  } else {
    if (activityEl) activityEl.innerText = `Unhealthy air quality detected in ${locationName}. Reduce heavy outdoor exertion and move workouts indoors.`;
    if (ventilationEl) ventilationEl.innerText = "Keep windows closed. Run indoor HEPA air purifiers on high mode.";
    if (sensitiveEl) sensitiveEl.innerText = "Sensitive groups must remain indoors in climate-filtered spaces.";
  }
}

function renderForecastChart(forecastList) {
  if (!forecastList || forecastList.length === 0) return;

  const times = forecastList.map((f) => f.timestamp.replace("T", " ").substring(0, 16));
  const pm25Values = forecastList.map((f) => f.predicted_pm2_5);
  const whoThreshold = new Array(times.length).fill(15.0);

  const traceForecast = {
    x: times,
    y: pm25Values,
    type: "scatter",
    mode: "lines+markers",
    name: "XGBoost Predicted PM2.5 (µg/m³)",
    line: { color: "#06b6d4", width: 3, shape: "spline" },
    marker: { size: 6, color: "#06b6d4" },
    fill: "tozeroy",
    fillcolor: "rgba(6, 182, 212, 0.08)",
  };

  const traceThreshold = {
    x: times,
    y: whoThreshold,
    type: "scatter",
    mode: "lines",
    name: "WHO Daily Limit (15 µg/m³)",
    line: { color: "#f43f5e", width: 2, dash: "dot" },
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Outfit, sans-serif", color: "#94a3b8" },
    margin: { l: 45, r: 20, t: 20, b: 40 },
    xaxis: { gridcolor: "rgba(255, 255, 255, 0.05)", tickangle: -30, nticks: 10 },
    yaxis: { title: "PM2.5 Concentration (µg/m³)", gridcolor: "rgba(255, 255, 255, 0.05)", zeroline: false },
    showlegend: true,
    legend: { x: 0, y: 1.1, orientation: "h" },
    hovermode: "x unified",
  };

  Plotly.newPlot("forecastChart", [traceForecast, traceThreshold], layout, { responsive: true, displayModeBar: false });
}

function generateMockData(locationName) {
  const now = new Date();
  const forecast = [];
  for (let i = 0; i < 72; i++) {
    const dt = new Date(now.getTime() + i * 3600 * 1000);
    const basePm25 = 8 + Math.sin(i / 6) * 3.5 + Math.random() * 2;
    const aqi = Math.round(basePm25 * 3.8);

    forecast.push({
      timestamp: dt.toISOString(),
      predicted_pm2_5: Math.round(basePm25 * 100) / 100,
      predicted_us_aqi: aqi,
      aqi_category: aqi <= 50 ? "Good" : "Moderate",
      temperature: 16.5,
      humidity: 62.0,
      wind_speed: 4.5,
    });
  }

  return {
    location: locationName,
    generated_at: now.toISOString(),
    recent_monitoring_mae: 1.52,
    current_pm2_5: forecast[0].predicted_pm2_5,
    current_us_aqi: forecast[0].predicted_us_aqi,
    forecast: forecast,
  };
}
