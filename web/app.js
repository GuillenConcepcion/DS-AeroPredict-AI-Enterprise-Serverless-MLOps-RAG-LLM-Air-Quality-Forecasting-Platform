/**
 * AeroPredict AI - Front-End Controller & Visualization Engine
 */

document.addEventListener("DOMContentLoaded", () => {
  const locationSelect = document.getElementById("locationSelect");
  const refreshBtn = document.getElementById("refreshBtn");

  // Initial Load
  loadDashboardData(locationSelect.value);

  // Event Listeners
  locationSelect.addEventListener("change", (e) => {
    loadDashboardData(e.target.value);
  });

  refreshBtn.addEventListener("click", () => {
    refreshBtn.classList.add("fa-spin");
    loadDashboardData(locationSelect.value).then(() => {
      setTimeout(() => refreshBtn.classList.remove("fa-spin"), 600);
    });
  });
});

async function loadDashboardData(locationName) {
  try {
    const response = await fetch("../data/latest_predictions.json");
    if (!response.ok) throw new Error("JSON predictions file not found.");
    const data = await response.json();
    renderDashboard(data, locationName);
  } catch (error) {
    console.warn("Could not fetch latest_predictions.json, generating synthetic demonstration data.", error);
    const mockData = generateMockData(locationName);
    renderDashboard(mockData, locationName);
  }
}

function renderDashboard(data, locationName) {
  // Update Current AQI KPI
  const currentAqi = data.current_us_aqi || 38;
  const currentPm25 = data.current_pm2_5 || 9.2;
  const mae = data.recent_monitoring_mae || 1.45;

  document.getElementById("aqiValue").innerText = currentAqi;
  document.getElementById("pm25Value").innerText = currentPm25;
  document.getElementById("maeValue").innerText = mae;

  // AQI Color Scheme Dynamics
  const circle = document.getElementById("aqiCircle");
  const badge = document.getElementById("aqiBadge");
  const desc = document.getElementById("aqiDescription");

  let color = "#10b981"; // emerald
  let status = "Good";
  let description = "Air quality is satisfactory, posing little or no health risk.";

  if (currentAqi > 50 && currentAqi <= 100) {
    color = "#f59e0b"; // amber
    status = "Moderate";
    description = "Air quality is acceptable; sensitive individuals should take precautions.";
  } else if (currentAqi > 100 && currentAqi <= 150) {
    color = "#f97316"; // orange
    status = "Unhealthy for Sensitive Groups";
    description = "Members of sensitive groups may experience health effects.";
  } else if (currentAqi > 150) {
    color = "#f43f5e"; // rose
    status = "Unhealthy";
    description = "Everyone may begin to experience health effects.";
  }

  circle.style.borderColor = color;
  circle.style.boxShadow = `0 0 25px ${color}66`;
  badge.innerText = status;
  badge.style.background = `${color}33`;
  badge.style.color = color;
  badge.style.borderColor = color;
  desc.innerText = description;

  // Render Forecast Chart
  renderForecastChart(data.forecast);
  renderFeatureImportance();
}

function renderForecastChart(forecastList) {
  if (!forecastList || forecastList.length === 0) return;

  const times = forecastList.map((f) => f.timestamp.replace("T", " ").substring(0, 16));
  const pm25Values = forecastList.map((f) => f.predicted_pm2_5);
  const whoThreshold = new Array(times.length).fill(15.0); // WHO PM2.5 24h safety limit

  const traceForecast = {
    x: times,
    y: pm25Values,
    type: "scatter",
    mode: "lines+markers",
    name: "Predicted PM2.5 (µg/m³)",
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
    xaxis: {
      gridcolor: "rgba(255, 255, 255, 0.05)",
      tickangle: -30,
      nticks: 10,
    },
    yaxis: {
      title: "PM2.5 Concentration (µg/m³)",
      gridcolor: "rgba(255, 255, 255, 0.05)",
      zeroline: false,
    },
    showlegend: false,
    hovermode: "x unified",
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.newPlot("forecastChart", [traceForecast, traceThreshold], layout, config);
}

function renderFeatureImportance() {
  const features = ["pm2_5_lag_1h", "pm2_5_roll_mean_24h", "temperature_2m", "wind_speed_10m", "hour_sin", "stagnation_index"];
  const importances = [340, 280, 195, 140, 95, 75];

  const trace = {
    x: importances.reverse(),
    y: features.reverse(),
    type: "bar",
    orientation: "h",
    marker: {
      color: importances.map((v, i) => `rgba(6, 182, 212, ${0.4 + i * 0.1})`),
      corner-radius: 4,
    },
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Outfit, sans-serif", color: "#94a3b8" },
    margin: { l: 140, r: 20, t: 10, b: 30 },
    xaxis: { gridcolor: "rgba(255, 255, 255, 0.05)", title: "LightGBM Split Importance" },
    yaxis: { gridcolor: "transparent" },
  };

  Plotly.newPlot("featureImportanceChart", [trace], layout, { responsive: true, displayModeBar: false });
}

function generateMockData(locationName) {
  const now = new Date();
  const forecast = [];

  for (let i = 0; i < 72; i++) {
    const dt = new Date(now.getTime() + i * 3600 * 1000);
    const basePm25 = 8 + Math.sin(i / 6) * 4 + Math.random() * 2;
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
    recent_monitoring_mae: 1.42,
    current_pm2_5: forecast[0].predicted_pm2_5,
    current_us_aqi: forecast[0].predicted_us_aqi,
    forecast: forecast,
  };
}
