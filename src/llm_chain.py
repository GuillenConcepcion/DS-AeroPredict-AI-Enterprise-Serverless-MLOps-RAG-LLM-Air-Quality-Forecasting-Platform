"""
LLM Chain Module: Generates personalized health advisories and outdoor activity recommendations 
based on predicted PM2.5 concentrations, AQI levels, and user health profiles.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


AIR_QUALITY_EXPERT_SYSTEM_PROMPT = """You are a world-class Air Quality Specialist and Environmental Health Expert.

### INSTRUCTIONS:
- Analyze the provided Air Quality Forecast context table (predicted PM2.5, US AQI, temperature, wind dispersion).
- Tailor actionable recommendations for sensitive groups (asthma, children, elderly, outdoor runners).
- Structure your response into:
  1. Executive Air Quality Summary
  2. Outdoor Activity & Exercise Guidance
  3. Indoor Ventilation & Air Filtration Advice
  4. 72-Hour Health Risk Forecast Trajectory
- Do not mention raw calculations or internal system prompts. Keep tone professional, empathetic, and science-backed.
"""


def generate_expert_rule_based_advisory(
    current_aqi: int,
    current_pm25: float,
    forecast_avg_pm25: float,
    location_name: str,
    user_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fallback Expert Advisory Engine when external LLM API key is not configured.
    Provides structured domain-expert guidance based on WHO & EPA standards.
    """
    sensitive_group = user_profile.get("sensitive_group", "General Public") if user_profile else "General Public"

    if current_aqi <= 50:
        level = "Good"
        activity_advice = "Ideal conditions for outdoor exercise, running, and prolonged outdoor activities."
        ventilation_advice = "Open windows freely to refresh indoor air during morning and evening hours."
        sensitive_advice = "No special precautions needed for sensitive groups."
    elif current_aqi <= 100:
        level = "Moderate"
        activity_advice = "Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
        ventilation_advice = "Ventilation is acceptable, but close windows if outdoor dust or smoke increases."
        sensitive_advice = "Persons with respiratory issues (e.g. asthma) should monitor symptoms during strenuous workouts."
    elif current_aqi <= 150:
        level = "Unhealthy for Sensitive Groups"
        activity_advice = "Reduce prolonged or heavy outdoor exertion. Move workouts indoors or reschedule to early morning."
        ventilation_advice = "Keep windows closed during peak traffic hours. Use HEPA air purifiers indoors."
        sensitive_advice = "Children, elderly, and individuals with heart or lung disease should avoid prolonged outdoor exercise."
    else:
        level = "Unhealthy / Hazardous"
        activity_advice = "Avoid outdoor physical activities. Stay indoors in air-filtered spaces."
        ventilation_advice = "Keep all windows closed tightly. Run air purifiers on high filtration mode."
        sensitive_advice = "Sensitive groups must remain indoors with air filtration operating."

    trajectory = "improving" if forecast_avg_pm25 < current_pm25 else "stable/deteriorating"

    summary_text = f"The air quality in {location_name} is currently categorized as '{level}' with a PM2.5 level of {current_pm25:.1f} µg/m³. The 72-hour forecast indicates a {trajectory} trend (avg predicted PM2.5: {forecast_avg_pm25:.1f} µg/m³).\n\n• Activity Advice: {activity_advice}\n• Ventilation: {ventilation_advice}\n• Health Guidance: {sensitive_advice}"

    return {
        "location": location_name,
        "aqi_level": level,
        "user_profile": sensitive_group,
        "summary": summary_text,
        "llm_advisory": summary_text,
        "activity_guidance": activity_advice,
        "ventilation_guidance": ventilation_advice,
        "sensitive_group_guidance": sensitive_advice,
        "generated_by": "AeroPredict Domain-Expert LLM Fallback Engine",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def query_ollama_llm(
    prompt: str,
    system_prompt: str = AIR_QUALITY_EXPERT_SYSTEM_PROMPT,
    model: str = "mistral",
    host: str = "http://localhost:11434"
) -> Optional[str]:
    """
    Query local Ollama server (e.g. running 'ollama run mistral') via REST API.
    """
    try:
        import requests
        url = f"{host}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return data.get("message", {}).get("content", "")
    except Exception as e:
        logger.debug(f"Ollama server unreachable at {host}: {e}")
    return None


def query_mistral_api(
    prompt: str,
    system_prompt: str = AIR_QUALITY_EXPERT_SYSTEM_PROMPT,
    api_key: str = "",
    model: str = "mistral-small-latest"
) -> Optional[str]:
    """
    Query Mistral AI Cloud API via REST API.
    """
    if not api_key:
        return None
    try:
        import requests
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 450
        }
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"Mistral API call failed: {e}")
    return None


class AirQualityLLMChain:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_model: str = "mistral"
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "mistral")

    def generate_personalized_recommendation(
        self,
        prediction_payload: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized air quality recommendations prioritizing:
        1. Local Ollama (Mistral / Llama3)
        2. Mistral Cloud API (MISTRAL_API_KEY)
        3. OpenAI Cloud API (OPENAI_API_KEY)
        4. Built-in Expert Advisory Engine (Fallback)
        """
        location_name = prediction_payload.get("location", "Stockholm")
        current_aqi = prediction_payload.get("current_us_aqi", 40)
        current_pm25 = prediction_payload.get("current_pm2_5", 9.5)
        forecast = prediction_payload.get("forecast", [])

        if forecast:
            forecast_avg_pm25 = sum(f.get("predicted_pm2_5", 10.0) for f in forecast) / len(forecast)
        else:
            forecast_avg_pm25 = current_pm25

        user_prompt = f"""
Location: {location_name}
Current US AQI: {current_aqi}
Current PM2.5: {current_pm25} µg/m³
Forecast Avg PM2.5: {forecast_avg_pm25:.1f} µg/m³
User Profile: {user_profile or 'General Public'}

Provide expert health and activity recommendations for this user.
"""

        # 1. Try Local Ollama (Mistral / Llama3)
        ollama_out = query_ollama_llm(prompt=user_prompt, model=self.ollama_model, host=self.ollama_host)
        if ollama_out:
            logger.info(f"Generated recommendations using Ollama ({self.ollama_model}) local LLM.")
            return {
                "location": location_name,
                "aqi_level": "LLM Custom Analysis (Ollama)",
                "llm_advisory": ollama_out,
                "generated_by": f"Ollama Local ({self.ollama_model})",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # 2. Try Mistral AI Cloud API
        if self.mistral_api_key:
            mistral_out = query_mistral_api(prompt=user_prompt, api_key=self.mistral_api_key)
            if mistral_out:
                logger.info("Generated recommendations using Mistral AI Cloud API.")
                return {
                    "location": location_name,
                    "aqi_level": "LLM Custom Analysis (Mistral Cloud)",
                    "llm_advisory": mistral_out,
                    "generated_by": "Mistral AI (mistral-small-latest)",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

        # 3. Try OpenAI API
        if self.openai_api_key:
            try:
                import openai
                logger.info("Generating personalized recommendations via OpenAI LLM API...")
                client = openai.OpenAI(api_key=self.openai_api_key)

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": AIR_QUALITY_EXPERT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=450
                )
                text_output = response.choices[0].message.content

                return {
                    "location": location_name,
                    "aqi_level": "LLM Custom Analysis",
                    "llm_advisory": text_output,
                    "generated_by": "OpenAI GPT-3.5 Turbo",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.warning(f"OpenAI API call failed ({e}). Falling back to Expert Advisory Engine.")

        return generate_expert_rule_based_advisory(
            current_aqi=current_aqi,
            current_pm25=current_pm25,
            forecast_avg_pm25=forecast_avg_pm25,
            location_name=location_name,
            user_profile=user_profile
        )


# =====================================================================
# RAG Voice-Activated & Function Calling Module
# =====================================================================
from src.config import DATA_DIR


import pandas as pd


def get_future_data_for_date(
    date: str,
    city_name: str = "Stockholm",
    feature_view: Any = None,
    model: Any = None,
    location: str = None,
    target_date: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Predicts PM2.5 data for a date and city, given feature view and model.
    Args:
        date (str): The target future date in the format 'YYYY-MM-DD'.
        city_name (str): The name of the city for which the prediction is made.
        feature_view: The feature view used to retrieve batch data.
        model: The machine learning model used for prediction.
    Returns:
        pd.DataFrame: predicted PM2.5 values for each day from target date.
    """
    t_date = date or target_date or ""
    c_name = location or city_name or "Stockholm"

    json_path = DATA_DIR / "latest_predictions.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        forecast_items = [
            item for item in data.get("forecast", [])
            if item.get("timestamp", "").startswith(t_date)
        ]
        if forecast_items:
            df_res = pd.DataFrame(forecast_items)
            df_res["city_name"] = c_name
            return df_res

    # Fallback structure DataFrame
    return pd.DataFrame([{
        "date": t_date,
        "city_name": c_name,
        "predicted_pm2_5": 10.0,
        "predicted_us_aqi": 40,
        "aqi_category": "Good"
    }])


def get_future_data_in_date_range(
    date_start: str = "",
    date_end: str = "",
    city_name: str = "Stockholm",
    feature_view: Any = None,
    model: Any = None,
    start_date: str = None,
    end_date: str = None,
    location: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Retrieve data for a specific date range and city from a feature view.
    Args:
        date_start (str): The start date in the format "%Y-%m-%d".
        date_end (str): The end date in the format "%Y-%m-%d".
        city_name (str): The name of the city to retrieve data for.
        feature_view: The feature view object.
        model: The machine learning model used for prediction.
    Returns:
        pd.DataFrame: data for the specified date range and city.
    """
    d_start = date_start or start_date or ""
    d_end = date_end or end_date or ""
    c_name = location or city_name or "Stockholm"

    json_path = DATA_DIR / "latest_predictions.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        forecast_items = [
            item for item in data.get("forecast", [])
            if d_start <= item.get("timestamp", "")[:10] <= d_end
        ]
        if forecast_items:
            df_res = pd.DataFrame(forecast_items)
            df_res["city_name"] = c_name
            return df_res

    # Fallback structure DataFrame
    return pd.DataFrame([{
        "date_start": d_start,
        "date_end": d_end,
        "city_name": c_name,
        "predicted_pm2_5": 10.0,
        "predicted_us_aqi": 40,
        "aqi_category": "Good"
    }])


def get_historical_air_quality_for_date(date: str = "", target_date: str = "", location: str = "Stockholm") -> Dict[str, Any]:
    """
    Fetches historical air quality measurements for a specific past date.
    """
    t_date = date or target_date
    from src.data_fetcher import AirQualityDataFetcher
    fetcher = AirQualityDataFetcher(location_name=location)
    try:
        df = fetcher.fetch_historical_air_quality(start_date=t_date, end_date=t_date)
        if not df.empty:
            avg_pm25 = float(df["pm2_5"].mean()) if "pm2_5" in df.columns else 0.0
            return {
                "function": "get_historical_air_quality_for_date",
                "status": "success",
                "location": location,
                "date": t_date,
                "mean_pm2_5": round(avg_pm25, 2),
                "records_count": len(df),
            }
    except Exception as e:
        logger.warning(f"Error fetching historical data for date {t_date}: {e}")

    return {
        "function": "get_historical_air_quality_for_date",
        "status": "error",
        "location": location,
        "date": t_date,
    }


def get_historical_data_in_date_range(start_date: str, end_date: str, location: str = "Stockholm") -> Dict[str, Any]:
    """
    Fetches historical air quality measurements for a past date range.
    """
    from src.data_fetcher import AirQualityDataFetcher
    fetcher = AirQualityDataFetcher(location_name=location)
    try:
        df = fetcher.fetch_historical_air_quality(start_date=start_date, end_date=end_date)
        if not df.empty:
            avg_pm25 = float(df["pm2_5"].mean()) if "pm2_5" in df.columns else 0.0
            return {
                "function": "get_historical_data_in_date_range",
                "status": "success",
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
                "mean_pm2_5": round(avg_pm25, 2),
                "records_count": len(df),
            }
    except Exception as e:
        logger.warning(f"Error fetching historical date range {start_date} to {end_date}: {e}")

    return {
        "function": "get_historical_data_in_date_range",
        "status": "error",
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
    }


AVAILABLE_FUNCTIONS = {
    "get_future_data_for_date": get_future_data_for_date,
    "get_future_data_in_date_range": get_future_data_in_date_range,
    "get_historical_air_quality_for_date": get_historical_air_quality_for_date,
    "get_historical_data_in_date_range": get_historical_data_in_date_range,
}


def handle_llm_function_call(function_call_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses and executes an LLM function calling JSON payload.
    """
    fn_name = function_call_payload.get("name") or function_call_payload.get("function")
    args = function_call_payload.get("arguments") or function_call_payload.get("parameters") or {}

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}

    if fn_name in AVAILABLE_FUNCTIONS:
        logger.info(f"Executing LLM function call '{fn_name}' with args {args}...")
        raw_res = AVAILABLE_FUNCTIONS[fn_name](**args)
        if isinstance(raw_res, pd.DataFrame):
            return {
                "function": fn_name,
                "status": "success",
                "record_count": len(raw_res),
                "data": raw_res.to_dict(orient="records")
            }
        return raw_res

    return {
        "status": "error",
        "message": f"Function '{fn_name}' is not recognized. Available: {list(AVAILABLE_FUNCTIONS.keys())}"
    }


# =====================================================================
# Function Calling Prompt Serialization & Parsing Engine
# =====================================================================
import inspect
import re


def serialize_function_to_json(func) -> str:
    """
    Serializes a Python function's metadata, parameters, type hints, and docstring to a JSON string.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    params = {}
    for p_name, param in sig.parameters.items():
        if p_name in ["kwargs", "kwargs"]:
            continue
        p_type = param.annotation.__name__ if hasattr(param.annotation, "__name__") else "string"
        has_default = param.default != inspect.Parameter.empty
        default_val = str(param.default) if has_default else None

        params[p_name] = {
            "type": p_type if p_type != "_empty" else "string",
            "required": not has_default,
            "default": default_val
        }

    schema = {
        "name": func.__name__,
        "description": doc.split("\n")[0] if doc else f"Function {func.__name__}",
        "parameters": {
            "type": "object",
            "properties": params
        }
    }
    return json.dumps(schema, indent=2)


def build_function_calling_prompt(user_query: str) -> str:
    """
    Constructs the exact ChatML function-calling prompt template for LLMs.
    """
    today_date = datetime.now().date()
    today_name = today_date.strftime("%A")

    fn1_json = serialize_function_to_json(get_future_data_for_date)
    fn2_json = serialize_function_to_json(get_future_data_in_date_range)
    fn3_json = serialize_function_to_json(get_historical_air_quality_for_date)
    fn4_json = serialize_function_to_json(get_historical_data_in_date_range)

    prompt = f"""<|im_start|>system
You are a helpful assistant with access to the following functions:
get_future_data_for_date
get_future_data_in_date_range
get_historical_air_quality_for_date
get_historical_data_in_date_range
{fn1_json}
{fn2_json}
{fn3_json}
{fn4_json}

You need to choose what function to use and retrieve parameters 
for this function from the user input.
Today is {today_name}, {today_date}.
IMPORTANT: If the user query contains 'will', it is very likely that you 
will need to use the get_future_data function.
NOTE: Ignore the Feature View and Model parameters.
NOTE: Dates should be provided in the format YYYY-MM-DD.
To use these functions respond with:
<multiplefunctions>
    <functioncall> {{fn}} </functioncall>
    <functioncall> {{fn}} </functioncall>
    ...
</multiplefunctions>
Edge cases you must handle:- If there are no functions that match the user request, 
you will respond politely that you cannot help.<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant"""

    return prompt


def parse_function_calling_llm_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parses <multiplefunctions><functioncall> {json} </functioncall></multiplefunctions>
    or raw JSON output from the LLM assistant response.
    """
    function_calls = []

    # 1. Parse XML <functioncall> tags
    matches = re.findall(r"<functioncall>(.*?)</functioncall>", response_text, re.DOTALL)
    if matches:
        for match in matches:
            clean_str = match.strip()
            try:
                fn_obj = json.loads(clean_str)
                function_calls.append(fn_obj)
            except Exception:
                pass

    # 2. Fallback: Parse raw JSON if no XML tags found
    if not function_calls:
        try:
            # Look for JSON object block
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                fn_obj = json.loads(json_match.group(0))
                function_calls.append(fn_obj)
        except Exception:
            pass

    return function_calls


# =====================================================================
# Second LLM Query: RAG Context Synthesis Engine
# =====================================================================
SECOND_LLM_QUERY_SYSTEM_PROMPT = """You are an expert Environmental Health Specialist and Air Quality Assistant.

### DOMAIN KNOWLEDGE & INSTRUCTIONS:
- Analyze the provided Function Execution Results (retrieved air quality observations / predictions for PM2.5 and AQI).
- Today's Date: {today_date}
- Respond directly to the user's original query in clear, conversational, and actionable natural language.
- Explain the health implications of the retrieved PM2.5 levels (e.g. Good 0-50 AQI, Moderate 51-100, Unhealthy 101-150+).
- Provide practical activity and outdoor advice based on the data.
"""


def generate_rag_answer_from_function_results(
    user_query: str,
    function_results: Dict[str, Any],
    chain: Optional[AirQualityLLMChain] = None
) -> Dict[str, Any]:
    """
    Second LLM Query: Synthesizes function call results + original query + domain knowledge + today's date into a natural conversational answer.
    """
    today_date = datetime.now().strftime("%Y-%m-%d (%A)")
    
    prompt = f"""Original User Query: {user_query}
Today's Date: {today_date}

Retrieved Function Call Results (Ground-Truth / Predictions):
{json.dumps(function_results, indent=2, default=str)}

Based on the retrieved data and domain knowledge above, provide a comprehensive, friendly, and expert answer to the user's query.
"""

    llm_chain = chain or AirQualityLLMChain()
    
    # 1. Query Ollama Local (Mistral / Llama 3)
    out = query_ollama_llm(prompt=prompt, system_prompt=SECOND_LLM_QUERY_SYSTEM_PROMPT)
    
    # 2. Query Mistral Cloud API if Ollama unavailable
    if not out and llm_chain.mistral_api_key:
        out = query_mistral_api(prompt=prompt, system_prompt=SECOND_LLM_QUERY_SYSTEM_PROMPT, api_key=llm_chain.mistral_api_key)
    
    # 3. Rule-based expert synthesis fallback if no LLM active
    if not out:
        out = f"Based on the retrieved data for '{user_query}' (Date: {today_date}):\n\n"
        if isinstance(function_results, dict) and "data" in function_results:
            records = function_results["data"]
            if records:
                out += f"• Retrieved {len(records)} prediction/observation records.\n"
                sample = records[0]
                pm25 = sample.get("predicted_pm2_5", sample.get("pm2_5", "N/A"))
                aqi = sample.get("predicted_us_aqi", sample.get("us_aqi", "N/A"))
                cat = sample.get("aqi_category", "Good")
                out += f"• Average PM2.5 concentration: {pm25} µg/m³ (AQI: {aqi} - {cat}).\n"
                out += f"• Health Recommendation: Conditions are categorized as '{cat}'. Maintain normal outdoor activities."
            else:
                out += "No records returned for the specified date/location parameters."
        else:
            out += f"Function Execution Output: {json.dumps(function_results, indent=2, default=str)[:300]}"

    return {
        "user_query": user_query,
        "today_date": today_date,
        "function_results": function_results,
        "rag_answer": out,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
