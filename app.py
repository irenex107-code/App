import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI


load_dotenv("api_key.env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "")

PROFILE_FILE = "user_profiles_demo.xlsx"
CONVERSATION_DIR = "conversations"
SURVEY_DIR = "user_surveys"
# Reply: main user-facing answer. Fast: classification, risk JSON, memory compression (lower latency).
OPENAI_MODEL_REPLY = os.environ.get("OPENAI_MODEL_REPLY", "gpt-5-mini")
OPENAI_MODEL_FAST = os.environ.get("OPENAI_MODEL_FAST", "gpt-4o-mini")
GROK_MODEL_REPLY = os.environ.get("GROK_MODEL_REPLY", "grok-4-fast-non-reasoning")
GROK_MODEL_FAST = os.environ.get("GROK_MODEL_FAST", "grok-3-mini")
TRUSTED_DOMAINS = [
    "who.int",
    "cdc.gov",
    "nih.gov",
    "nimh.nih.gov",
    "medlineplus.gov",
    "nhs.uk",
    "mayoclinic.org",
    "apa.org",
]
MAX_RAW_MESSAGES = 24
KEEP_RECENT_MESSAGES = 10


def ensure_demo_profile_excel(path: str = PROFILE_FILE) -> None:
    """Create a demo Excel profile file if missing."""
    if os.path.exists(path):
        return

    demo_data = [
        {
            "user_id": "U1001",
            "name": "Alex",
            "age": 24,
            "gender": "female",
            "occupation": "graduate student",
            "stressors": "academic pressure, sleep disruption",
            "support_system": "2 close friends, sibling",
            "mental_health_history": "mild anxiety episodes",
            "preferences": "short practical suggestions, breathing exercises",
            "risk_level": "low",
        },
        {
            "user_id": "U1002",
            "name": "Ben",
            "age": 31,
            "gender": "male",
            "occupation": "software engineer",
            "stressors": "workload, perfectionism",
            "support_system": "partner, weekly sports group",
            "mental_health_history": "burnout last year",
            "preferences": "structured plans and CBT-style reframing",
            "risk_level": "medium",
        },
        {
            "user_id": "U1003",
            "name": "Cathy",
            "age": 42,
            "gender": "female",
            "occupation": "teacher",
            "stressors": "caregiving and family conflict",
            "support_system": "limited local support",
            "mental_health_history": "none diagnosed",
            "preferences": "empathetic language and journaling prompts",
            "risk_level": "low",
        },
    ]
    df = pd.DataFrame(demo_data)
    df.to_excel(path, index=False)


def ensure_conversation_dir(path: str = CONVERSATION_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_survey_dir(path: str = SURVEY_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_name_key(name: str) -> str:
    """Stable filesystem-safe key from display name (matches save and load)."""
    name = (name or "").strip()
    if not name:
        return "anonymous"
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if ascii_slug:
        return ascii_slug[:120]
    return "u_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def survey_path(name_key: str) -> str:
    return os.path.join(SURVEY_DIR, f"{name_key}.json")


def save_user_survey(form: Dict[str, Any]) -> str:
    """Validate, persist questionnaire JSON. Returns name_key used for storage and chat."""
    ensure_survey_dir()
    display_name = str(form.get("name", "")).strip()
    if not display_name:
        raise ValueError("请填写姓名")
    name_key = normalize_name_key(display_name)
    record: Dict[str, Any] = {
        "user_id": name_key,
        "name": display_name,
        "age": form.get("age"),
        "gender": form.get("gender") or "",
        "occupation": form.get("occupation") or "",
        "stressors": form.get("stressors") or "",
        "support_system": form.get("support_system") or "",
        "mental_health_history": form.get("mental_health_history") or "",
        "preferences": form.get("preferences") or "",
        "risk_level": form.get("risk_level") or "low",
        "saved_at": datetime.utcnow().isoformat(),
    }
    path = survey_path(name_key)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return name_key


def load_survey_by_name(name: str) -> Optional[Dict[str, Any]]:
    name_key = normalize_name_key(name)
    path = survey_path(name_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def profile_from_survey_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Same shape as Excel-derived profile for the chat pipeline."""
    keys = [
        "user_id",
        "name",
        "age",
        "gender",
        "occupation",
        "stressors",
        "support_system",
        "mental_health_history",
        "preferences",
        "risk_level",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        v = rec.get(k, "")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = ""
        out[k] = v
    return out


@st.cache_data
def load_profiles(path: str = PROFILE_FILE) -> pd.DataFrame:
    return pd.read_excel(path)


def call_llm(provider: str, system_prompt: str, user_prompt: str, *, fast: bool = False) -> str:
    """Unified LLM call for OpenAI/Grok. Use fast=True for short JSON/summary tasks (smaller models)."""
    try:
        if provider == "GPT (OpenAI)":
            if not OPENAI_API_KEY:
                return "OpenAI API key missing. Please set OPENAI_API_KEY in api_key.env."
            model = OPENAI_MODEL_FAST if fast else OPENAI_MODEL_REPLY
            llm = ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=0)
            result = llm.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            return getattr(result, "content", str(result))

        if not GROK_API_KEY:
            return "Grok API key missing. Please set GROK_API_KEY in api_key.env."
        model = GROK_MODEL_FAST if fast else GROK_MODEL_REPLY
        grok_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
        response = grok_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"Model invocation failed: {exc}"


def conversation_path(user_id: str) -> str:
    return os.path.join(CONVERSATION_DIR, f"{user_id}.json")


def default_messages() -> List[Dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "Hi, I'm here to support you. Share how you're feeling today, and I'll provide "
                "personalized coping suggestions."
            ),
            "time": datetime.utcnow().isoformat(),
        }
    ]


def load_conversation(user_id: str) -> List[Dict[str, str]]:
    path = conversation_path(user_id)
    if not os.path.exists(path):
        return default_messages()
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    return default_messages()


def save_conversation(user_id: str, messages: List[Dict[str, str]]) -> None:
    path = conversation_path(user_id)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=True, indent=2)


def append_message(user_id: str, role: str, content: str) -> None:
    messages = load_conversation(user_id)
    messages.append({"role": role, "content": content, "time": datetime.utcnow().isoformat()})
    save_conversation(user_id, messages)


def recent_memory_window(messages: List[Dict[str, str]], n: int = 8) -> List[Dict[str, str]]:
    summary_msgs = [m for m in messages if m.get("role") == "memory_summary"]
    raw_msgs = [m for m in messages if m.get("role") != "memory_summary"]
    memory_slice = raw_msgs[-n:]
    if summary_msgs:
        return [summary_msgs[-1]] + memory_slice
    return memory_slice


def summarize_messages(
    provider: str,
    existing_summary: str,
    msgs_to_summarize: List[Dict[str, str]],
) -> str:
    if not msgs_to_summarize:
        return existing_summary
    system_prompt = """
You are a conversation memory compressor for a mental-health support chatbot.
Create a concise persistent memory summary for future turns.
Rules:
- Keep it factual and brief.
- Include: user background cues, recurring stressors, emotional patterns, helpful interventions, unresolved concerns.
- Do not include unnecessary details.
- Keep under 180 words.
Return plain text only.
"""
    payload = {
        "existing_summary": existing_summary,
        "messages_to_compress": msgs_to_summarize,
    }
    result = call_llm(provider, system_prompt, json.dumps(payload, ensure_ascii=True), fast=True)
    return result.strip() if result.strip() else existing_summary


def compress_conversation_if_needed(user_id: str, provider: str) -> None:
    messages = load_conversation(user_id)
    summary_msgs = [m for m in messages if m.get("role") == "memory_summary"]
    raw_msgs = [m for m in messages if m.get("role") != "memory_summary"]
    if len(raw_msgs) <= MAX_RAW_MESSAGES:
        return

    existing_summary = summary_msgs[-1]["content"] if summary_msgs else ""
    to_compress = raw_msgs[:-KEEP_RECENT_MESSAGES]
    recent_raw = raw_msgs[-KEEP_RECENT_MESSAGES:]
    new_summary = summarize_messages(provider, existing_summary, to_compress)
    summary_message = {
        "role": "memory_summary",
        "content": new_summary,
        "time": datetime.utcnow().isoformat(),
    }
    save_conversation(user_id, [summary_message] + recent_raw)


def detect_emotion_and_intent(provider: str, user_text: str) -> Dict[str, Any]:
    """Preprocessing module: emotion + intent recognition."""
    system_prompt = """
You are an emotion and intent classifier for mental-health chat support.
Output only JSON (no markdown), with this schema:
{
  "emotion": "one short label",
  "intensity": 0-10,
  "intent": "one short label",
  "risk_flags": ["optional list of risk signals"],
  "analysis_notes": "one concise sentence"
}
"""
    user_prompt = f"User message:\n{user_text}"
    raw = call_llm(provider, system_prompt, user_prompt, fast=True).strip()

    try:
        parsed = json.loads(raw)
        parsed.setdefault("emotion", "unknown")
        parsed.setdefault("intensity", 5)
        parsed.setdefault("intent", "general_support")
        parsed.setdefault("risk_flags", [])
        parsed.setdefault("analysis_notes", "")
        return parsed
    except Exception:
        return {
            "emotion": "unknown",
            "intensity": 5,
            "intent": "general_support",
            "risk_flags": ["parse_failure"],
            "analysis_notes": f"Raw model output: {raw[:300]}",
        }


def keyword_risk_scan(user_text: str) -> Dict[str, Any]:
    text = user_text.lower()
    self_harm_keywords = [
        "suicide", "kill myself", "end my life", "self harm", "cut myself", "don't want to live",
    ]
    harm_others_keywords = [
        "hurt someone", "kill him", "kill her", "kill them", "attack someone", "violent thoughts",
    ]
    crisis_keywords = [
        "right now", "tonight", "goodbye forever", "final message", "last message", "no reason to live",
    ]

    self_hits = [k for k in self_harm_keywords if k in text]
    other_hits = [k for k in harm_others_keywords if k in text]
    crisis_hits = [k for k in crisis_keywords if k in text]

    if crisis_hits and (self_hits or other_hits):
        level = "crisis"
    elif self_hits or other_hits:
        level = "high"
    elif crisis_hits:
        level = "medium"
    else:
        level = "low"

    return {
        "rule_level": level,
        "self_harm_hits": self_hits,
        "harm_others_hits": other_hits,
        "crisis_hits": crisis_hits,
    }


def detect_emotion_intent_and_llm_risk_combined(
    provider: str, user_text: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Single LLM call for emotion/intent + LLM risk triage (saves one round trip vs two calls)."""
    system_prompt = """
You are an emotion/intent classifier and clinical risk triage assistant for mental-health chat support.
Output only JSON (no markdown), with this combined schema:
{
  "emotion": "one short label",
  "intensity": 0-10,
  "intent": "one short label",
  "risk_flags": ["optional list of risk signals"],
  "analysis_notes": "one concise sentence",
  "risk_level": "low|medium|high|crisis",
  "risk_type": ["self_harm","harm_others","acute_distress","none"],
  "rationale": "one short sentence",
  "safety_actions": ["short action list"]
}
"""
    user_prompt = f"User message:\n{user_text}"
    raw = call_llm(provider, system_prompt, user_prompt, fast=True).strip()
    try:
        parsed = json.loads(raw)
        emotion_info = {
            "emotion": parsed.get("emotion", "unknown"),
            "intensity": parsed.get("intensity", 5),
            "intent": parsed.get("intent", "general_support"),
            "risk_flags": parsed.get("risk_flags", []),
            "analysis_notes": parsed.get("analysis_notes", ""),
        }
        llm_risk = {
            "risk_level": parsed.get("risk_level", "low"),
            "risk_type": parsed.get("risk_type", ["none"]),
            "rationale": parsed.get("rationale", ""),
            "safety_actions": parsed.get("safety_actions", []),
        }
        return emotion_info, llm_risk
    except Exception:
        return (
            {
                "emotion": "unknown",
                "intensity": 5,
                "intent": "general_support",
                "risk_flags": ["parse_failure"],
                "analysis_notes": f"Raw model output: {raw[:300]}",
            },
            {
                "risk_level": "medium",
                "risk_type": ["acute_distress"],
                "rationale": f"Failed to parse combined JSON: {raw[:120]}",
                "safety_actions": ["Ask supportive follow-up and provide crisis resources when needed."],
            },
        )


def llm_risk_assessment(provider: str, user_text: str, emotion_info: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = """
You are a clinical risk triage assistant for a mental-health chatbot.
Output only JSON:
{
  "risk_level": "low|medium|high|crisis",
  "risk_type": ["self_harm","harm_others","acute_distress","none"],
  "rationale": "one short sentence",
  "safety_actions": ["short action list"]
}
"""
    user_prompt = (
        f"User text: {user_text}\n"
        f"Emotion-intent context: {json.dumps(emotion_info, ensure_ascii=True)}"
    )
    raw = call_llm(provider, system_prompt, user_prompt, fast=True).strip()
    try:
        parsed = json.loads(raw)
        parsed.setdefault("risk_level", "low")
        parsed.setdefault("risk_type", ["none"])
        parsed.setdefault("rationale", "")
        parsed.setdefault("safety_actions", [])
        return parsed
    except Exception:
        return {
            "risk_level": "medium",
            "risk_type": ["acute_distress"],
            "rationale": f"Failed to parse risk JSON: {raw[:120]}",
            "safety_actions": ["Ask supportive follow-up and provide crisis resources when needed."],
        }


def merge_risk_assessment(rule_risk: Dict[str, Any], llm_risk: Dict[str, Any]) -> Dict[str, Any]:
    levels = {"low": 0, "medium": 1, "high": 2, "crisis": 3}
    reverse = {v: k for k, v in levels.items()}
    final_level_num = max(levels.get(rule_risk["rule_level"], 0), levels.get(llm_risk.get("risk_level", "low"), 0))
    final_level = reverse.get(final_level_num, "low")
    return {
        "final_risk_level": final_level,
        "rule_based": rule_risk,
        "llm_based": llm_risk,
        "requires_crisis_message": final_level in {"high", "crisis"},
    }


def extract_url_from_duckduckgo_link(link: str) -> str:
    parsed = urlparse(link)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query_dict = parse_qs(parsed.query)
        uddg = query_dict.get("uddg", [""])[0]
        return uddg or link
    return link


def domain_in_whitelist(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return any(netloc == d or netloc.endswith(f".{d}") for d in TRUSTED_DOMAINS)


def fetch_page_metadata(url: str) -> Dict[str, str]:
    try:
        res = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        text = res.text[:50000]
        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        desc_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else url
        snippet = re.sub(r"\s+", " ", desc_match.group(1)).strip() if desc_match else "No summary available."
        return {"title": title[:160], "snippet": snippet[:300], "url": url}
    except Exception:
        return {"title": url, "snippet": "Metadata unavailable.", "url": url}


def fetch_metadata_parallel(urls: List[str]) -> List[Dict[str, str]]:
    if not urls:
        return []
    workers = min(5, len(urls))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fetch_page_metadata, urls))


# Search runs in parallel with the first LLM call; emotion labels are not known yet, so use neutral defaults.
DEFAULT_SEARCH_EMOTION_HINT: Dict[str, Any] = {"emotion": "stress", "intent": "support"}


def search_web_context(query: str, emotion_info: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, str]]:
    """Search module: whitelist trusted medical sources and return traceable citations."""
    search_query = (
        f"mental health coping strategies {emotion_info.get('emotion', 'stress')} "
        f"{emotion_info.get('intent', 'support')} {query} for {profile.get('occupation', 'adult')} "
        "site:who.int OR site:cdc.gov OR site:nih.gov OR site:nimh.nih.gov OR site:medlineplus.gov "
        "OR site:nhs.uk OR site:mayoclinic.org OR site:apa.org"
    )

    try:
        res = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": search_query},
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        res.raise_for_status()
        html = res.text
    except Exception as exc:
        return [{"title": "Search unavailable", "snippet": str(exc), "url": ""}]

    links = re.findall(r'href="(https?://[^"]+)"', html)
    unique_urls: List[str] = []
    for link in links:
        clean_url = extract_url_from_duckduckgo_link(link)
        if clean_url.startswith("http") and domain_in_whitelist(clean_url):
            if clean_url not in unique_urls:
                unique_urls.append(clean_url)
        if len(unique_urls) >= 5:
            break

    if not unique_urls:
        return [
            {
                "title": "No trusted-source hits",
                "snippet": "No whitelist source found for this query.",
                "url": "",
            }
        ]
    return fetch_metadata_parallel(unique_urls)


def build_final_response(
    provider: str,
    user_text: str,
    profile: Dict[str, Any],
    emotion_info: Dict[str, Any],
    risk_info: Dict[str, Any],
    web_results: List[Dict[str, str]],
    memory_messages: List[Dict[str, str]],
) -> str:
    """Answer module: personalized response using profile + emotion + web context."""
    system_prompt = """
You are a careful, empathetic mental health chatbot.
- Give supportive, non-judgmental, practical guidance.
- Personalize suggestions based on profile and detected emotion/intent.
- Use web context if relevant, but do not fabricate facts.
- Cite source claims inline like [1], [2] when using web context.
- Keep response concise and structured:
  1) Empathy
  2) Practical steps (3-5 bullet points)
  3) One gentle reflection question
  4) If risk signs are present, include a brief safety recommendation.
- Do NOT claim to be a doctor.
- If the user has imminent self-harm or harm intent, recommend immediate emergency/crisis support.
"""

    user_prompt = (
        f"User profile:\n{json.dumps(profile, ensure_ascii=True)}\n\n"
        f"Emotion/Intent result:\n{json.dumps(emotion_info, ensure_ascii=True)}\n\n"
        f"Risk assessment:\n{json.dumps(risk_info, ensure_ascii=True)}\n\n"
        f"Conversation memory (recent turns):\n{json.dumps(memory_messages, ensure_ascii=True)}\n\n"
        f"Web search results:\n{json.dumps(web_results, ensure_ascii=True)}\n\n"
        f"User message:\n{user_text}"
    )
    return call_llm(provider, system_prompt, user_prompt)


def format_references(web_results: List[Dict[str, str]]) -> str:
    refs = [r for r in web_results if r.get("url")]
    if not refs:
        return ""
    lines = ["\n\nReferences:"]
    for idx, item in enumerate(refs, start=1):
        lines.append(f"- [{idx}] {item.get('title', 'Source')} - {item['url']}")
    return "\n".join(lines)


st.set_page_config(page_title="Mental Health Chatbot Demo", page_icon="🧠", layout="wide")
st.title("🧠 Mental Health Chatbot (Demo)")
st.caption(
    "问卷保存档案 → 按姓名加载 → 画像驱动对话：合并情绪/风险（1 次 LLM）∥ 可信来源检索 → 最终回复。"
)

ensure_demo_profile_excel()
ensure_survey_dir()
ensure_conversation_dir()

with st.sidebar:
    st.header("Settings")
    model_provider = st.selectbox("Model Provider", ["GPT (OpenAI)", "Grok (xAI)"])
    show_pipeline_data = st.checkbox("Show pipeline details", value=True)
    enable_web_search = st.checkbox(
        "Enable trusted web search",
        value=True,
        help="Turn off to skip DuckDuckGo + page fetches (much faster; replies use profile/memory only).",
    )

    if model_provider == "GPT (OpenAI)":
        st.info(f"OpenAI — reply: `{OPENAI_MODEL_REPLY}` · fast: `{OPENAI_MODEL_FAST}`")
    else:
        st.info(f"Grok — reply: `{GROK_MODEL_REPLY}` · fast: `{GROK_MODEL_FAST}`")

tab_survey, tab_chat = st.tabs(["1. 调查问卷", "2. 对话"])

with tab_survey:
    st.subheader("请填写问卷（将按姓名保存）")
    with st.form("intake_survey"):
        q_name = st.text_input("姓名 *", placeholder="与后续登录对话时一致")
        q_age = st.number_input("年龄", min_value=0, max_value=120, value=25, step=1)
        q_gender = st.selectbox("性别", ["", "female", "male", "non-binary", "prefer not to say"])
        q_occupation = st.text_input("职业 / 身份")
        q_stressors = st.text_area("主要压力来源")
        q_support = st.text_area("支持系统（家人、朋友等）")
        q_history = st.text_area("心理健康相关背景（可选）")
        q_prefs = st.text_area("偏好（例如：希望回答简短、偏好某种技巧）")
        q_risk = st.selectbox("自评当前风险等级（用于系统提示，非诊断）", ["low", "medium", "high"])
        submitted = st.form_submit_button("提交并保存")
    if submitted:
        try:
            key = save_user_survey(
                {
                    "name": q_name,
                    "age": int(q_age),
                    "gender": q_gender,
                    "occupation": q_occupation,
                    "stressors": q_stressors,
                    "support_system": q_support,
                    "mental_health_history": q_history,
                    "preferences": q_prefs,
                    "risk_level": q_risk,
                }
            )
            st.success(f"已保存。姓名键：`{key}`。请到「对话」页输入相同姓名并点击加载。")
            st.session_state["last_survey_name"] = q_name.strip()
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"保存失败：{exc}")

with tab_chat:
    st.subheader("加载问卷档案")
    default_name = st.session_state.get("last_survey_name", "")
    c1, c2 = st.columns([3, 1])
    with c1:
        lookup_name = st.text_input(
            "请输入姓名（须与问卷一致）",
            value=default_name,
            key="lookup_name_input",
            placeholder="与问卷中填写的姓名一致",
        )
    with c2:
        load_clicked = st.button("加载档案", type="primary")

    if load_clicked and lookup_name.strip():
        rec = load_survey_by_name(lookup_name)
        if rec is None:
            st.error("未找到该姓名对应的问卷，请先完成「调查问卷」或检查姓名拼写。")
        else:
            st.session_state["active_profile"] = profile_from_survey_record(rec)
            st.session_state["chat_user_id"] = str(rec.get("user_id", normalize_name_key(lookup_name)))
            st.success(f"已加载：{rec.get('name', lookup_name)}")
    elif load_clicked:
        st.warning("请输入姓名。")

    selected_profile = st.session_state.get("active_profile")
    user_id = st.session_state.get("chat_user_id")

    if selected_profile and user_id:
        with st.expander("当前用户画像（来自已保存问卷）"):
            st.json(selected_profile)

        current_uid = st.session_state.get("profile_conversation_uid")
        if current_uid != user_id:
            st.session_state["profile_conversation_uid"] = user_id
            st.session_state["messages"] = load_conversation(user_id)

        for msg in st.session_state["messages"]:
            if msg["role"] == "memory_summary":
                continue
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Tell me what is on your mind..."):
            user_msg = {"role": "user", "content": prompt, "time": datetime.utcnow().isoformat()}
            st.session_state["messages"].append(user_msg)
            append_message(user_id, "user", prompt)
            st.chat_message("user").write(prompt)

            with st.spinner("Running analysis + web search (parallel)..."):
                rule_risk = keyword_risk_scan(prompt)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_llm = executor.submit(
                        detect_emotion_intent_and_llm_risk_combined, model_provider, prompt
                    )
                    fut_web = None
                    if enable_web_search:
                        fut_web = executor.submit(
                            search_web_context, prompt, DEFAULT_SEARCH_EMOTION_HINT, selected_profile
                        )
                    emotion_intent, llm_risk = fut_llm.result()
                    web_context = fut_web.result() if fut_web else []

                risk_summary = merge_risk_assessment(rule_risk, llm_risk)

            with st.spinner("Generating final personalized response..."):
                final_reply = build_final_response(
                    model_provider,
                    prompt,
                    selected_profile,
                    emotion_intent,
                    risk_summary,
                    web_context,
                    recent_memory_window(st.session_state["messages"]),
                )
                final_reply = final_reply + format_references(web_context)
                if risk_summary.get("requires_crisis_message"):
                    final_reply += (
                        "\n\nIf you might act on thoughts of harming yourself or others, please contact your local "
                        "emergency number immediately or your nearest crisis hotline."
                    )

            assistant_msg = {"role": "assistant", "content": final_reply, "time": datetime.utcnow().isoformat()}
            st.session_state["messages"].append(assistant_msg)
            append_message(user_id, "assistant", final_reply)
            compress_conversation_if_needed(user_id, model_provider)
            st.session_state["messages"] = load_conversation(user_id)
            st.chat_message("assistant").write(final_reply)

            if show_pipeline_data:
                with st.expander("Pipeline outputs"):
                    st.subheader("1) Emotion + Intent")
                    st.json(emotion_intent)
                    st.subheader("2) Risk Triage")
                    st.json(risk_summary)
                    st.subheader("3) Trusted Web Search Results")
                    st.json(web_context)
    else:
        st.info("请先在「调查问卷」填写并保存，然后在本页输入**相同姓名**并点击「加载档案」后再开始对话。")
