import os
import streamlit as st

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage


# ========= 1. 读取环境变量中的 API Key =========
from dotenv import load_dotenv

load_dotenv("api_key.env")  # 自动从 .env 文件读取环境变量

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROK_API_KEY = os.environ["GROK_API_KEY"]


# ========= 2. 搭建 LangChain Prompt 模板 =========
template = ChatPromptTemplate.from_messages(
    [
        # System Prompt
        SystemMessagePromptTemplate.from_template(
            """
You are a professional nutrition analyst. The user will provide the name of a food item. For each input, complete the following tasks:

1. Provide a detailed nutritional analysis of the food, including:
   - Basic nutritional elements: calories, protein, fat, carbohydrates, dietary fiber, sugar, etc.
   - Key micronutrients: vitamins (A, B, C, D, E), minerals (calcium, iron, potassium, magnesium, sodium, zinc, selenium, etc.).

2. Give a brief 1–2 sentence explanation of the function or health impact of each nutrient.

3. Based on the overall nutritional value and health impact (e.g., sugar level, salt level, fat level, degree of processing, presence of beneficial nutrients), give a 
   **health score from 0 to 100**, along with a one-sentence explanation.

4. Output using the following structure:

[Food Name]
- Basic Nutritional Elements:
  - …
- Micronutrients:
  - …
- Health Score (0–100): xx
- Score Explanation: xxxx

If the user input is not a food name, politely ask the user to provide a valid food item.
"""
        ),

        # Example 1
        HumanMessage(content="Apple"),
        AIMessage(
            content="""
[Food Name] Apple

Basic Nutritional Elements:

Calories: ~52 kcal per 100g — Low-calorie, suitable as a healthy snack.

Carbohydrates: 14g — Mostly natural fructose, provides quick energy.

Dietary Fiber: 2.4g — Supports digestion and gut health.

Protein: 0.3g — Minimal amount.

Fat: 0.2g — Very low fat.

Sugar: ~10g — Naturally occurring fruit sugar.

Micronutrients:

Vitamin C: 4.6mg — Supports immunity and acts as an antioxidant.

Potassium: 107mg — Helps regulate blood pressure and heart function.

Vitamin A (small amount) — Supports eye health.

Iron (trace amount) — Important for red blood cell function.

Health Score (0–100): 89

Score Explanation:
Apple is low in fat and calories, rich in fiber and antioxidants; the only drawback is its natural sugar content.
"""
        ),

        # Example 2
        HumanMessage(content="Fried chicken"),
        AIMessage(
            content="""
[Food Name] Fried Chicken

Basic Nutritional Elements:

Calories: ~260–300 kcal per 100g — High calorie density.

Fat: 15–20g — Contains significant saturated fat, which is not ideal for heart health.

Protein: 12–16g — Good protein source despite being fried.

Carbohydrates: 8–12g — Mainly from breading and frying process.

Sodium: 600–900mg — Very high salt level, increases blood pressure risk.

Micronutrients:

Iron: 0.8–1mg — Supports red blood cell production.

Potassium (moderate level) — Beneficial but overshadowed by high sodium.

B Vitamins (from chicken meat) — Important for energy metabolism.

Limited amounts of other vitamins and minerals due to frying.

Health Score (0–100): 42

Score Explanation:
Although high in protein, fried chicken is calorie-dense, high in fat and sodium, and involves an unhealthy cooking method, resulting in a low health score.
"""
        ),

        # 最后一条是带变量的 Human 模板
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)


# ========= 3. 初始化模型 =========
chat_llm = ChatOpenAI(
    model="gpt-5-mini",   # 可根据你账号实际模型名称调整
    api_key=OPENAI_API_KEY,
    temperature=0,
)

# xAI Grok 客户端
client_grok = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)


def LLM_invoke(food_name: str):
    """调用 OpenAI + LangChain 模板进行营养分析"""
    messages = template.format_messages(text=food_name)
    try:
        result = chat_llm.invoke(messages)
        return getattr(result, "content", result)
    except Exception as e:
        return f"Model invocation failed: {e}"


# ========= 4. Streamlit UI 部分 =========
st.title("🍎 Nutrition Analyzer Chatbot")

# Sidebar for model selection
with st.sidebar:
    st.title("🔧 Settings")
    model_provider = st.selectbox(
        "Select Language Model Provider",
        ["OpenAI", "xAI Grok"],
    )

    if model_provider == "OpenAI":
        st.info("Using OpenAI model: gpt-5-mini")
        model_name = "gpt-5-mini"
    else:
        st.info("Using xAI model: grok-4")
        model_name = "grok-4-fast-non-reasoning"


# 初始化对话历史（只用于前端展示）
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Tell me a food name, and I'll analyze its nutrition for you."}
    ]

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# 输入与响应
if prompt := st.chat_input("Enter a food name, e.g., Banana, Pizza..."):
    # 记录用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # -------- 根据模型提供方生成回复 --------
    if model_provider == "OpenAI":
        # 用你上面定义的 LLM_invoke + 模板
        msg = LLM_invoke(prompt)

    else:  # xAI Grok
        # 给 Grok 一个等价的 system prompt，每次只根据当前输入回答
        grok_system_prompt = """
You are a professional nutrition analyst. The user will provide the name of a food item.
For each input, do a detailed nutrition analysis and output with the following structure:

[Food Name]
- Basic Nutritional Elements:
  - …
- Micronutrients:
  - …
- Health Score (0–100): xx
- Score Explanation: xxxx

If the user input is not a food name, politely ask the user to provide a valid food item.
"""

        try:
            response = client_grok.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": grok_system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            msg = response.choices[0].message.content
        except Exception as e:
            msg = f"Grok model invocation failed: {e}"

    # 记录并显示助手回复
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
