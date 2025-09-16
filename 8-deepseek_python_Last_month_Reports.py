import re
import os
import logging
from datetime import datetime
from difflib import get_close_matches
from functools import lru_cache
from io import BytesIO
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# =========================
# Load environment variables
# =========================
load_dotenv()

TOKEN = os.getenv("TOKEN")
FILE = os.getenv("FILE_PATH")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TOKEN:
    raise ValueError("❌ TOKEN not found in .env file")
if not FILE:
    raise ValueError("❌ FILE_PATH not found in .env file")

# =========================
# Setup Logging to File
# =========================
logging.basicConfig(
    filename="bot.log",
    filemode="a",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# =========================
# Google Gemini Setup
# =========================
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    logger.info("🧠 Gemini AI Agent is ready.")
else:
    gemini_model = None
    logger.warning("⚠️ Gemini API Key not found. AI features disabled.")

def ask_gemini(prompt: str) -> str:
    """Ask Gemini and return response"""
    if not gemini_model:
        return "❌ ميزة الذكاء الاصطناعي غير متاحة حاليًا."

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"❌ خطأ في Gemini: {e}")
        return "عذرًا، تعذر الاتصال بالذكاء الاصطناعي."

# =========================
# Helpers
# =========================
def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("'", "'").replace("'", "'").replace("`", "'")
    s = s.lower()
    s = re.sub(r"[^0-9a-z\u0600-\u06FF\s'&\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def format_num(value: float) -> str:
    """Format number to B, M, K with 2 decimal places"""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"

def fuzzy_pick(text: str, choices_norm: list, cutoff=0.6):
    t = norm(text)
    if not t:
        return None
    for c in choices_norm:
        if c and c in t:
            return c
    toks = t.split()
    for n in range(min(4, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            chunk = " ".join(toks[i:i+n])
            matches = get_close_matches(chunk, choices_norm, n=1, cutoff=cutoff)
            if matches:
                return matches[0]
    return None

# =========================
# Load data (once)
# =========================
if not os.path.exists(FILE):
    logger.error(f"❌ File not found: {FILE}")
    exit(1)

try:
    df = pd.read_csv(FILE, dtype=str)
    logger.info(f"✅ Loaded CSV: {len(df)} rows")
except Exception as e:
    logger.error(f"❌ Failed to load CSV: {e}")
    exit(1)

# =========================
# Data Cleaning & Validation (Safe & Non-Intrusive)
# =========================
logger.info("🧹 Starting safe data cleaning...")

# 1. تنظيف أسماء الأعمدة
df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]

# 2. تأكد من الأعمدة الأساسية
required_cols = ["Service", "Operator", "YTD Actual"]
for col in required_cols:
    if col not in df.columns:
        logger.error(f"❌ Missing required column: {col}")
        exit(1)

# 3. تنظيف الأعمدة العددية
# في قسم تنظيف الأعمدة العددية (النقطة 3)
for col in ["YTD Actual", "YTD Budget", "FY Budget"]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.replace(" ", ""),  # تغيير ast إلى astype
            errors="coerce"
        ).fillna(0.0)

# وفي قسم تنظيف Year (النقطة 4)
if "Year" in df.columns:
    df["Year"] = pd.to_numeric(
        df["Year"].astype(str).str.extract(r"(20\d{2})", expand=False),  # تغيير ast إلى astype
        errors="coerce"
    )
    df = df[df["Year"].notna()]
    df["Year"] = df["Year"].astype(int)
    df = df[(df["Year"] >= 2020) & (df["Year"] <= 2030)]

# Add Month handling if column exists
if "Month" in df.columns:
    # Clean month column - handle numeric or text months
    df["Month"] = df["Month"].fillna("")
    
    # Create a mapping for month names to numbers
    month_map = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
        'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
        'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    
    def parse_month(month_str):
        month_str = str(month_str).lower().strip()
        # Try to convert to int directly
        try:
            month_num = int(month_str)
            if 1 <= month_num <= 12:
                return month_num
        except:
            pass
        # Try to map month name
        for month_name, month_num in month_map.items():
            if month_name in month_str:
                return month_num
        return None
    
    df["Month_Num"] = df["Month"].apply(parse_month)
    
    # Create Year-Month column for easier grouping
    df["Year_Month"] = df.apply(
        lambda x: f"{int(x['Year'])}-{int(x['Month_Num']):02d}" 
        if pd.notna(x.get('Month_Num')) and pd.notna(x.get('Year')) 
        else None, axis=1
    )

# 5. تنظيف Service و Operator
df["_svc_norm"] = df["Service"].fillna("").apply(norm)
df["_op_norm"] = df["Operator"].fillna("").apply(norm)

# 6. احذف السطور الفارغة
df = df[(df["_svc_norm"] != "") & (df["_op_norm"] != "")]
df = df.dropna(subset=["YTD Actual"])

# 7. احذف السطور المكررة تمامًا
df = df.drop_duplicates()

# 8. احذف السطور اللي فيها "Total Revenues" كخدمة (لأنها نتيجة مجمعة مش بيانات أولية)
invalid_svc_keywords = ["total", "all", "overall", "revenue", "revenues", "summary", "consolidated"]
df = df[~df["_svc_norm"].str.contains('|'.join(invalid_svc_keywords), na=False, case=False)]

logger.info(f"✅ Cleaned  {len(df)} rows remaining")

# =========================
# Operator mapping
# =========================
OP_GROUPS = {
    "vodafone": ["vodafone", "vodafone data", "vfe"],
    "etisalat": ["etisalat", "egy net", "egy net / eitsalat 2021", "nol",
                 "egy net / admin. fees", "etm", "etisal"],
    "orange": ["orange", "orange data", "oeg"],
    "others": ["others 2021", "noor", "other", "vsat", "mena", "others",
               "noor / mena", "ncts", "yalla misr"]
}

MNO_ISP_MAPPING = {
    "vodafone": "MNO's",
    "etisalat": "MNO's",
    "orange": "MNO's",
    "others": "ISP's"
}

ALL_OP_ALIASES = {a for lst in OP_GROUPS.values() for a in lst}
OP_CANON_FROM_ALIAS = {alias: grp for grp, lst in OP_GROUPS.items() for alias in lst}

# =========================
# Services hierarchy
# =========================
SERVICES_HIERARCHY = {
    "infrastructure": {
        "transmission": {
            "mno's transmission": {},
            "tx iru": {},
            "isp's transmission": {}
        },
        "international": {
            "ipt": {},
            "iplcs": {},
            "data center": {},
            "cross connection": {}
        },
        "colocation": {
            "isp's colocation": {},
            "mno's colocation": {}
        }
    },
    "access": {
        "local loop": {},
        "adsl": {},
        "bit stream": {},
        "ftts": {},
        "fiber access": {}
    },
    "voice": {
        "int'l og": {},
        "fvno": {},
        "itfs": {},
        "rnr": {}
    }
}

def flatten_services(hierarchy, parent=None):
    flat = {}
    for service, children in hierarchy.items():
        full_path = f"{parent} {service}".strip() if parent else service
        flat[service] = full_path
        flat.update(flatten_services(children, full_path))
    return flat

FLAT_SERVICES = flatten_services(SERVICES_HIERARCHY)

# =========================
# Service Synonyms
# =========================
SERVICE_SYNONYMS = {
    "total": "total revenues",
    "total revenue": "total revenues",
    "total revenues": "total revenues",
    "revenue": "total revenues",
    "revenues": "total revenues",
    "all": "total revenues",
    "overall": "total revenues",
    "infra": "infrastructure",
    "inf": "infrastructure",
    "trans": "transmission",
    "tx": "transmission",
    "intl": "international",
    "colo": "colocation",
    "data": "data center",
    "cross": "cross connection",
    "loop": "local loop",
    "fiber": "fiber access",
    "og": "int'l og",
    "international outgoing": "int'l og",
    "bit": "bit stream",
    "adsl": "adsl",
    "ftts": "ftts",
    "voice": "voice",
    "access": "access"
}

# =========================
# User Last Query
# =========================
user_last_query = {}

# =========================
# Query Logging (CSV)
# =========================
import csv
from collections import Counter

# إنشاء ملف CSV مع الهيدر إذا كان أول مرة
if not os.path.exists("user_queries.csv"):
    with open("user_queries.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "User ID", "Username", "First Name", "Text"])

def log_query(user_id: int, username: str, first_name: str, text: str):
    """Log user query to CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("user_queries.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_id, username, first_name, text])

# قائمة لتحليل الأسئلة الشائعة
query_log = []

# =========================
# Core logic
# =========================
def detect_year(text: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        return int(m.group(1))
    if "حتى الآن" in text or "currently" in text or "now" in text:
        return int(df["Year"].max()) if "Year" in df.columns and df["Year"].max() > 0 else None
    return None

def detect_operator(text: str) -> tuple[str | None, str | None, str | None]:
    t = norm(text)
    for canonical, aliases in OP_GROUPS.items():
        for alias in aliases:
            if alias in t:
                return canonical, alias, MNO_ISP_MAPPING.get(canonical, "Unknown")
    pick = fuzzy_pick(t, list(ALL_OP_ALIASES), cutoff=0.7)
    if pick:
        canonical = OP_CANON_FROM_ALIAS[pick]
        return canonical, pick, MNO_ISP_MAPPING.get(canonical, "Unknown")
    return None, None, None

def detect_service(text: str) -> tuple[str | None, str | None]:
    t = norm(text)
    for syn, service in SERVICE_SYNONYMS.items():
        if syn in t:
            return service, FLAT_SERVICES.get(service, service)
    for service, full_path in FLAT_SERVICES.items():
        if service in t:
            return service, full_path
    pick = fuzzy_pick(t, list(FLAT_SERVICES.keys()), cutoff=0.7)
    if pick:
        return pick, FLAT_SERVICES[pick]
    return None, None

def get_service_data(base_df: pd.DataFrame, service_norm: str) -> pd.DataFrame:
    """Get service data including children services"""
    def get_all_children(service_key):
        return [
            service for service, full_path in FLAT_SERVICES.items()
            if service_key in full_path and service != service_key
        ]
    
    children = get_all_children(service_norm)
    if children:
        result_df = pd.DataFrame()
        for child in children:
            child_rows = base_df[base_df["_svc_norm"] == child]
            result_df = pd.concat([result_df, child_rows])
        return result_df.drop_duplicates() if not result_df.empty else result_df
    else:
        service_df = base_df[base_df["_svc_norm"] == service_norm]
        return service_df.drop_duplicates() if not service_df.empty else service_df

def get_contribution_summary(year=None, op_canonical=None):
    """احسب مساهمة كل خدمة (بالمئة) بناءً على السنة والمشغل"""
    q = df.copy()
    if year: q = q[q["Year"] == year]
    if op_canonical:
        aliases = set(OP_GROUPS[op_canonical])
        q = q[q["_op_norm"].isin(aliases)]

    services = ["infrastructure", "access", "voice"]
    contributions = {}
    total = 0.0

    for svc in services:
        data = get_service_data(q, svc)
        amount = data["YTD Actual"].sum()
        contributions[svc] = amount
        total += amount

    if total == 0:
        return None

    # حولها لنسب مئوية
    result = {}
    for svc, val in contributions.items():
        result[svc] = {"value": val, "percent": (val / total * 100) if total != 0 else 0}

    return result, total

@lru_cache(maxsize=128)
def compute_answer(text: str) -> str:
    year = detect_year(text)
    op_canonical, op_match, op_type = detect_operator(text)
    service_norm, service_full = detect_service(text)

    if year is None:
        year = df["Year"].max() if "Year" in df.columns else None

    q = df.copy()
    if year and "Year" in q.columns:
        q = q[q["Year"] == year]

    if op_canonical:
        aliases = set(OP_GROUPS[op_canonical])
        q = q[q["_op_norm"].isin(aliases)]
        op_label = op_match.capitalize()
    else:
        op_label = "All Operators"

    if service_norm == "total revenues":
        total_actual = total_ytd_budget = total_fy_budget = 0.0
        for svc in ["infrastructure", "access", "voice"]:
            data = get_service_data(q, svc)
            total_actual += data["YTD Actual"].sum()
            total_ytd_budget += data["YTD Budget"].sum()
            total_fy_budget += data["FY Budget"].sum()

        var = total_actual - total_ytd_budget
        var_pct = (var / total_ytd_budget * 100) if total_ytd_budget != 0 else 0
        achievement = (total_actual / total_fy_budget * 100) if total_fy_budget != 0 else 0

        ytxt = f" {year}" if year else " (Latest)"
        reply = (
            f"📈 <b>Total Revenues{ytxt}</b>\n"
            f"👤 <b>Operator:</b> {op_label}\n"
            f"📞 <b>Type:</b> {op_type}\n"
            f"📊 <b>FY Budget:</b> {format_num(total_fy_budget)}\n"
            f"🎯 <b>YTD Budget:</b> {format_num(total_ytd_budget)}\n"
            f"✅ <b>YTD Actual:</b> {format_num(total_actual)}\n"
            f"📉 <b>VAR:</b> {format_num(var)}\n"
            f"📊 <b>Var %:</b> {var_pct:.1f}%\n"
            f"🏆 <b>Progress:</b> {achievement:.1f}% of FY"
        )
        return reply

    if not service_norm and not op_canonical:
        return (
            "❌ I couldn't detect a service or operator.\n"
            "Try: <code>Total Revenues Vodafone</code>, <code>Transmission 2025</code>, or <code>/help</code>"
        )

    if service_norm:
        service_rows = get_service_data(q, service_norm)
    else:
        service_rows = q

    if service_rows.empty:
        serv_name = service_full or "selection"
        ytxt = f" {year}" if year else ""
        otxt = f" for {op_label}" if op_label != "All Operators" else ""
        return f"❌ No data for {serv_name}{ytxt}{otxt}."

    actual = service_rows["YTD Actual"].sum()
    ytd_budget = service_rows["YTD Budget"].sum()
    fy_budget = service_rows["FY Budget"].sum()
    var = actual - ytd_budget
    var_pct = (var / ytd_budget * 100) if ytd_budget != 0 else 0
    achievement = (actual / fy_budget * 100) if fy_budget != 0 else 0

    serv_name = service_full or "Total Revenues"
    ytxt = f" {year}" if year else " (Latest)"

    reply = (
        f"📈 <b>{serv_name}{ytxt}</b>\n"
        f"👤 <b>Operator:</b> {op_label}\n"
        f"📞 <b>Type:</b> {op_type}\n"
        f"📊 <b>FY Budget:</b> {format_num(fy_budget)}\n"
        f"🎯 <b>YTD Budget:</b> {format_num(ytd_budget)}\n"
        f"✅ <b>YTD Actual:</b> {format_num(actual)}\n"
        f"📉 <b>VAR:</b> {format_num(var)}\n"
        f"📊 <b>Var %:</b> {var_pct:.1f}%\n"
        f"🏆 <b>Progress:</b> {achievement:.1f}% of FY"
    )
    return reply

# =========================
# Keyboards Setup
# =========================

# Main Menu Keyboard
def get_main_keyboard():
    """لوحة المفاتيح الرئيسية"""
    keyboard = [
        ["📊 Total Revenues", "📈 Service Trends"],
        ["📅 Monthly Trends", "🆚 Compare Operators"],
        ["📋 Summary Report", "🔧 All Services"],
        ["📄 Last Month Reports", "🤖 Smart AI"],
        ["📞 Help", "🔄 Last Query"]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

# Services Inline Keyboard (with Total Revenues)
def get_services_keyboard():
    """لوحة اختيار الخدمات"""
    keyboard = [
        [
            InlineKeyboardButton("💰 Total Revenues", callback_data="svc_total revenues"),
            InlineKeyboardButton("🏗️ Infrastructure", callback_data="svc_infrastructure")
        ],
        [
            InlineKeyboardButton("🌐 Access", callback_data="svc_access"),
            InlineKeyboardButton("📞 Voice", callback_data="svc_voice")
        ],
        [
            InlineKeyboardButton("📡 Transmission", callback_data="svc_transmission"),
            InlineKeyboardButton("🌍 International", callback_data="svc_international")
        ],
        [
            InlineKeyboardButton("🏢 Colocation", callback_data="svc_colocation"),
            InlineKeyboardButton("🔗 Local Loop", callback_data="svc_local loop")
        ],
        [
            InlineKeyboardButton("💻 ADSL", callback_data="svc_adsl"),
            InlineKeyboardButton("🌐 Bit Stream", callback_data="svc_bit stream")
        ],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(keyboard)

# Operators Inline Keyboard
def get_operators_keyboard():
    """لوحة اختيار المشغلين"""
    keyboard = [
        [
            InlineKeyboardButton("📱 Vodafone", callback_data="op_vodafone"),
            InlineKeyboardButton("🌟 Etisalat", callback_data="op_etisalat")
        ],
        [
            InlineKeyboardButton("🍊 Orange", callback_data="op_orange"),
            InlineKeyboardButton("👥 Others", callback_data="op_others")
        ],
        [
            InlineKeyboardButton("🌍 All Operators", callback_data="op_all"),
            InlineKeyboardButton("❌ Cancel", callback_data="cancel")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

# Years Inline Keyboard
def get_years_keyboard():
    """لوحة اختيار السنوات"""
    current_year = datetime.now().year
    years = list(range(2020, current_year + 1))

    keyboard = []
    row = []
    for i, year in enumerate(years):
        row.append(InlineKeyboardButton(str(year), callback_data=f"year_{year}"))
        if (i + 1) % 3 == 0:  # 3 years per row
            keyboard.append(row)
            row = []

    if row:  # Add remaining years
        keyboard.append(row)

    keyboard.append([
        InlineKeyboardButton("📅 Current Year", callback_data=f"year_{current_year}"),
        InlineKeyboardButton("❌ Cancel", callback_data="cancel")
    ])

    return InlineKeyboardMarkup(keyboard)

# Chart Types Keyboard
def get_chart_types_keyboard():
    """لوحة أنواع الرسوم البيانية"""
    keyboard = [
        [
            InlineKeyboardButton("📊 Bar Chart", callback_data="chart_bar"),
            InlineKeyboardButton("📈 Line Trend", callback_data="chart_line")
        ],
        [
            InlineKeyboardButton("🥧 Pie Chart", callback_data="chart_pie"),
            InlineKeyboardButton("🆚 Comparison", callback_data="chart_compare")
        ],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(keyboard)

# AI Quick Questions Keyboard
def get_ai_questions_keyboard():
    """أسئلة الذكاء الاصطناعي السريعة"""
    keyboard = [
        [InlineKeyboardButton("💰 What's the revenue breakdown?", callback_data="ai_breakdown")],
        [InlineKeyboardButton("📊 Which operator leads?", callback_data="ai_leader")],
        [InlineKeyboardButton("📈 How are the trends?", callback_data="ai_trends")],
        [InlineKeyboardButton("🎯 Budget vs Actual analysis?", callback_data="ai_budget")],
        [InlineKeyboardButton("🔍 Market share insights?", callback_data="ai_market")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(keyboard)

# =========================
# User Session Management
# =========================
user_sessions = {}

class UserSession:
    def __init__(self):
        self.current_action = None
        self.selected_service = None
        self.selected_operator = None
        self.selected_year = None
        self.chart_type = None
        self.first_operator = None

    def reset(self):
        self.__init__()

def get_user_session(user_id):
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession()
    return user_sessions[user_id]

# =========================
# Chart Generation Functions
# =========================
async def generate_pie_chart(query, year=None, operator=None):
    """Generate pie chart for services distribution"""
    q = df.copy()
    if year:
        q = q[q["Year"] == int(year)]
    if operator and operator != "all":
        q = q[q["_op_norm"].isin(OP_GROUPS.get(operator, []))]
    
    services_data = {}
    for svc in ["infrastructure", "access", "voice"]:
        data = get_service_data(q, svc)
        amount = data["YTD Actual"].sum()
        if amount > 0:
            services_data[svc.capitalize()] = amount
    
    if not services_data:
        await query.edit_message_text("❌ No data available for pie chart.")
        return
    
    plt.figure(figsize=(8, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    plt.pie(services_data.values(), labels=services_data.keys(), autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title(f"Revenue Distribution {year if year else 'All Years'}")
    plt.axis('equal')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"🥧 Revenue Distribution {'for ' + str(year) if year else ''}",
        parse_mode="HTML"
    )

async def generate_bar_chart(query, service=None, year=None, operator=None):
    """Generate bar chart"""
    q = df.copy()
    if year:
        q = q[q["Year"] == int(year)]
    
    if service:
        service_norm, service_full = detect_service(service)
        service_rows = get_service_data(q, service_norm)
        actual = service_rows["YTD Actual"].sum()
        budget = service_rows["YTD Budget"].sum()
        
        plt.figure(figsize=(6, 4))
        bars = plt.bar(["YTD Actual", "YTD Budget"], [actual, budget], color=["#AAACAB", "#9E049E"])
        plt.title(f"{service_full} | {year or 'All Years'}", fontsize=12)
        plt.ylabel("Revenue")
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(actual, budget)*0.02,
                     format_num(height), ha='center', fontsize=9)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
    else:
        # Compare main services
        total_infra = get_service_data(q, "infrastructure")["YTD Actual"].sum()
        total_access = get_service_data(q, "access")["YTD Actual"].sum()
        total_voice = get_service_data(q, "voice")["YTD Actual"].sum()
        
        plt.figure(figsize=(8, 5))
        services = ['Infrastructure', 'Access', 'Voice']
        values = [total_infra, total_access, total_voice]
        bars = plt.bar(services, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title(f"Services Comparison {year if year else 'All Years'}")
        plt.ylabel("Revenue")
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                     format_num(height), ha='center', fontsize=9)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    
    await query.message.reply_photo(
        photo=buf,
        caption="📊 Revenue Comparison",
        parse_mode="HTML"
    )

async def generate_comparison_chart(query, year=None):
    """Generate comparison chart between operators"""
    q = df.copy()
    if year:
        q = q[q["Year"] == int(year)]
    
    op_data = {}
    for op, aliases in OP_GROUPS.items():
        op_aliases = set(alias for alias in aliases)
        op_df = q[q["_op_norm"].apply(lambda x: any(alias in x for alias in op_aliases))]
        total = op_df["YTD Actual"].sum()
        if total > 0:
            op_data[op.capitalize()] = total
    
    if not op_data:
        await query.edit_message_text("❌ No data for comparison.")
        return
    
    plt.figure(figsize=(10, 6))
    operators = list(op_data.keys())
    values = list(op_data.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E77E']
    bars = plt.bar(operators, values, color=colors[:len(operators)])
    plt.title(f"Operators Comparison {year if year else 'All Years'}")
    plt.ylabel("Revenue")
    plt.xlabel("Operator")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                 format_num(height), ha='center', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    
    await query.message.reply_photo(
        photo=buf,
        caption=f"🆚 Operators Comparison {'for ' + str(year) if year else ''}",
        parse_mode="HTML"
    )

# =========================
# Fixed Callback Functions
# =========================
async def generate_trend_chart(service, year=None, operator=None, query=None, update=None):
    """Unified function to generate trend charts"""
    service_norm, service_full = detect_service(service)

    if not service_norm:
        error_msg = "❌ Service not found."
        if query:
            await query.edit_message_text(error_msg)
        elif update:
            await update.message.reply_text(error_msg, parse_mode="HTML")
        return

    q = df.copy()
    if year:
        q = q[q["Year"] == int(year)]
    if operator and operator != "all":
        q = q[q["_op_norm"].isin(OP_GROUPS.get(operator, []))]

    # Special handling for total revenues
    if service_norm == "total revenues":
        yearly_data = {}
        for y in q["Year"].unique():
            if y and y > 0:
                year_df = q[q["Year"] == y]
                total = 0
                for svc in ["infrastructure", "access", "voice"]:
                    service_data = get_service_data(year_df, svc)
                    total += service_data["YTD Actual"].sum()
                yearly_data[y] = total
    else:
        # Regular service handling
        yearly_data = {}
        for y in q["Year"].unique():
            if y and y > 0:
                year_df = q[q["Year"] == y]
                service_data = get_service_data(year_df, service_norm)
                yearly_data[y] = service_data["YTD Actual"].sum()

    if not yearly_data:
        error_msg = "❌ No data for this service over years."
        if query:
            await query.edit_message_text(error_msg)
        elif update:
            await update.message.reply_text(error_msg, parse_mode="HTML")
        return

    # Sort by year
    years = sorted(yearly_data.keys())
    values = [yearly_data[y] for y in years]

    # Create chart
    plt.figure(figsize=(10, 5))
    plt.plot(years, values, marker='o', color='blue', linewidth=2, markersize=6)

    # Add value labels
    for y, value in zip(years, values):
        plt.text(y, value + max(values)*0.02, format_num(value),
                ha='center', fontsize=9, color='blue')

    title_suffix = f" | {year}" if year else ""
    plt.title(f"Trend: {service_full}{title_suffix}", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Revenue")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    # Send the chart
    caption = f"📈 Trend for <b>{service_full}</b>"
    try:
        if query:
            await query.message.reply_photo(photo=buf, caption=caption, parse_mode="HTML")
            await query.edit_message_text(f"✅ تم إنشاء الرسم البياني لـ {service_full}")
        elif update:
            await update.message.reply_photo(photo=buf, caption=caption, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error sending trend chart: {e}")
        error_msg = f"❌ خطأ في إنشاء الرسم البياني: {str(e)}"
        if query:
            await query.edit_message_text(error_msg)
        elif update:
            await update.message.reply_text(error_msg, parse_mode="HTML")

async def monthly_trend_callback(query, context, service, year=None):
    """Generate monthly trend chart for a service"""
    service_norm, service_full = detect_service(service)

    if not service_norm:
        await query.edit_message_text("❌ Service not found.")
        return

    # Check if month data exists
    if "Year_Month" not in df.columns:
        await query.edit_message_text("❌ Monthly data not available in the dataset.")
        return

    q = df[df["Year_Month"].notna()].copy()

    # Filter by year if provided
    if year:
        q = q[q["Year"] == int(year)]
    
    # Special handling for total revenues
    if service_norm == "total revenues":
        monthly_data = {}
        for ym in q["Year_Month"].unique():
            month_df = q[q["Year_Month"] == ym]
            total = 0
            for svc in ["infrastructure", "access", "voice"]:
                service_data = get_service_data(month_df, svc)
                total += service_data["YTD Actual"].sum()
            monthly_data[ym] = total
    else:
        # Regular service handling
        monthly_data = {}
        for ym in q["Year_Month"].unique():
            month_df = q[q["Year_Month"] == ym]
            service_data = get_service_data(month_df, service_norm)
            value = service_data["YTD Actual"].sum()
            if value > 0:
                monthly_data[ym] = value
    
    if not monthly_data:
        await query.edit_message_text("❌ No monthly data for this service.")
        return

    # Sort by year-month
    months = sorted(monthly_data.keys())
    values = [monthly_data[m] for m in months]
    
    # Create chart
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(months)), values, marker='o', color='green', linewidth=2, markersize=5)
    
    # Add value labels (skip some for clarity if too many)
    step = max(1, len(months) // 12)
    for i in range(0, len(months), step):
        plt.text(i, values[i] + max(values)*0.02, format_num(values[i]), 
                ha='center', fontsize=8, color='green')
    
    plt.title(f"Monthly Trend: {service_full}", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(range(0, len(months), max(1, len(months)//12)), 
               [months[i] for i in range(0, len(months), max(1, len(months)//12))], 
               rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    # Send the chart
    try:
        await query.message.reply_photo(
            photo=buf,
            caption=f"📅 Monthly Trend for <b>{service_full}</b>",
            parse_mode="HTML"
        )
        await query.edit_message_text(f"✅ تم إنشاء الرسم البياني الشهري لـ {service_full}")
    except Exception as e:
        logger.error(f"Error sending monthly trend chart: {e}")
        await query.edit_message_text(f"❌ خطأ في إنشاء الرسم البياني: {str(e)}")
    
    # Reset session
    session = get_user_session(query.from_user.id)
    session.reset()

async def summary_callback(query, context, year):
    """Generate summary report for a year"""
    year = int(year) if year and str(year).isdigit() else None
    q = df.copy()
    if year:
        q = q[q["Year"] == year]

    data = {}
    for op, aliases in OP_GROUPS.items():
        op_aliases = set(alias for alias in aliases)
        op_df = q[q["_op_norm"].apply(lambda x: any(alias in x for alias in op_aliases))]
        total = op_df["YTD Actual"].sum()
        if total > 0:
            data[op] = total

    if not data:
        await query.edit_message_text("❌ No data found for any operator.")
        return

    lines = [f"📋 <b>Summary{' ' + str(year) if year else ''}</b>"]
    for op, val in sorted(data.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"• {op.capitalize()}: {format_num(val)}")
    
    await query.edit_message_text("\n".join(lines), parse_mode="HTML")
    
    # Reset session
    session = get_user_session(query.from_user.id)
    session.reset()

async def generate_chart_by_type(query, session, chart_type):
    """Generate different chart types"""
    service = session.selected_service
    operator = session.selected_operator
    year = session.selected_year
    
    if chart_type == "pie":
        await generate_pie_chart(query, year, operator)
    elif chart_type == "bar":
        await generate_bar_chart(query, service, year, operator)
    if chart_type == "line":
        if service:
            await generate_trend_chart(service, query=query)
        else:
            await query.edit_message_text("❌ Please select a service first.")
    elif chart_type == "compare":
        await generate_comparison_chart(query, year)
    
    session.reset()

async def handle_comparison(query, compare_type):
    """Handle different comparison types"""
    session = get_user_session(query.from_user.id)

    if compare_type == "all":
        await query.edit_message_text(
            "📅 اختر السنة للمقارنة:",
            reply_markup=get_years_keyboard()
        )
        session.current_action = "compare_all_year"
    elif compare_type == "two":
        await query.edit_message_text(
            "📅 اختر السنة للمقارنة:",
            reply_markup=get_years_keyboard()
        )
        session.current_action = "compare_two_year"
    elif compare_type == "service":
        await query.edit_message_text(
            "🔧 اختر الخدمة للمقارنة:",
            reply_markup=get_services_keyboard()
        )
        session.current_action = "compare_by_service"

async def generate_total_revenues_report(query, session):
    """إنشاء تقرير الإيرادات الإجمالية"""
    operator = session.selected_operator
    year = session.selected_year

    # Build query text
    query_text = f"Total Revenues {operator} {year}"

    try:
        reply = compute_answer(query_text)
        await query.edit_message_text(
            reply,
            parse_mode="HTML"
        )
    except Exception as e:
        await query.edit_message_text(f"❌ خطأ في إنشاء التقرير: {str(e)}")

    session.reset()

async def handle_ai_quick_question(query, ai_type):
    """التعامل مع الأسئلة السريعة للذكاء الاصطناعي"""
    questions = {
        "breakdown": "What is the revenue breakdown by service for the latest year?",
        "leader": "Which operator has the highest revenue and why?",
        "trends": "What are the main revenue trends over the past years?",
        "budget": "How is the actual performance compared to budget across services?",
        "market": "What insights can you provide about market share and competition?"
    }

    question = questions.get(ai_type, "Analyze the revenue data")

    await query.edit_message_text("🤖 جاري التحليل بالذكاء الاصطناعي...")

    if gemini_model:
        # Get context data
        year = df["Year"].max() if "Year" in df.columns else None
        contrib = get_contribution_summary(year, None)

        if contrib:
            contributions, total = contrib
            context_data = f"Total Revenue: {format_num(total)}\n"
            for svc, data in contributions.items():
                context_data += f"• {svc.title()}: {format_num(data['value'])} ({data['percent']:.1f}%)\n"
        else:
            context_data = "No revenue data available."

        prompt = f"""
        You are a telecom revenue analyst. Answer this question clearly and professionally in Arabic:

        Question: {question}

        Current Data Summary:
        {context_data}

        Provide insights, trends, and actionable recommendations.
        """

        ai_reply = ask_gemini(prompt)
        await query.edit_message_text(ai_reply)
    else:
        await query.edit_message_text("❌ ميزة الذكاء الاصطناعي غير متاحة حاليًا.")

# =========================
# Fixed Callback Handler
# =========================
async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """التعامل مع الأزرار المضغوطة"""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    session = get_user_session(user_id)
    data = query.data

    if data == "cancel":
        session.reset()
        await query.edit_message_text(
            "❌ تم الإلغاء. اختر من القائمة الرئيسية:",
            reply_markup=None
        )
        return

    # Handle service selection
    if data.startswith("svc_"):
        service = data[4:]  # Remove "svc_" prefix
        session.selected_service = service

        if session.current_action == "service_trends":
            # Show trend for selected service
            await query.edit_message_text(f"📈 جاري تحليل اتجاه {service}...")
            await generate_trend_chart(service, query=query)

        elif session.current_action == "monthly_trends_service":
            # Show operator selection for monthly trend
            await query.edit_message_text(
                f"🏢 اختر المشغل للاتجاه الشهري لـ {service}:",
                reply_markup=get_operators_keyboard()
            )
            session.current_action = "monthly_trends_operator"

        elif session.current_action == "chart_service":
            await query.edit_message_text(
                f"📊 اختر نوع الرسم البياني لـ {service}:",
                reply_markup=get_chart_types_keyboard()
            )

        elif session.current_action == "compare_by_service":
            # Show comparison for this service across operators
            await query.edit_message_text(f"🆚 جاري مقارنة {service} بين المشغلين...")
            await compare_service_operators(query, service)

        else:
            # Default: show service data
            reply = compute_answer(service)
            await query.edit_message_text(reply, parse_mode="HTML")

    # Handle operator selection
    elif data.startswith("op_"):
        operator = data[3:]  # Remove "op_" prefix
        session.selected_operator = operator

        if session.current_action == "total_revenues":
            if operator == "all":
                # Show all operators total
                reply = compute_answer("Total Revenues")
                await query.edit_message_text(reply, parse_mode="HTML")
                session.reset()
            else:
                await query.edit_message_text(
                    f"📅 اختر السنة لعرض إيرادات {operator}:",
                    reply_markup=get_years_keyboard()
                )

        elif session.current_action == "monthly_trends_operator":
            # Operator selected for monthly trends, now show year selection
            await query.edit_message_text(
                f"📅 اختر السنة للاتجاه الشهري لـ {session.selected_service} مع {operator}:",
                reply_markup=get_years_keyboard()
            )
            session.current_action = "monthly_trends_year"

        elif session.current_action == "compare_two_step1":
            # First operator selected, now ask for second
            session.current_action = "compare_two_step2"
            await query.edit_message_text(
                f"🔄 تم اختيار {operator}. اختر المشغل الثاني:",
                reply_markup=get_operators_keyboard()
            )

        elif session.current_action == "compare_two_step2":
            # Second operator selected, do comparison
            first_op = session.selected_operator
            await compare_two_operators(query, first_op, operator)
            session.reset()

    # Handle year selection
    elif data.startswith("year_"):
        year = data[5:]  # Remove "year_" prefix
        session.selected_year = year

        if session.current_action == "total_revenues":
            # Generate total revenues report
            await generate_total_revenues_report(query, session)

        elif session.current_action == "summary":
            # Generate summary report
            await summary_callback(query, context, year)

        elif session.current_action == "monthly_trends_year":
            # Generate monthly trend for selected service and year
            await monthly_trend_callback(query, context, session.selected_service, year)

        elif session.current_action == "compare_all_year":
            # Generate comparison for all operators for this year
            await generate_comparison_chart(query, year)
            session.reset()

    # Handle AI quick questions
    elif data.startswith("ai_"):
        ai_type = data[3:]  # Remove "ai_" prefix
        await handle_ai_quick_question(query, ai_type)

    # Handle chart types
    elif data.startswith("chart_"):
        chart_type = data[6:]  # Remove "chart_" prefix
        session.chart_type = chart_type
        await generate_chart_by_type(query, session, chart_type)

    # Handle comparison types
    elif data.startswith("compare_"):
        compare_type = data[8:]  # Remove "compare_" prefix
        await handle_comparison(query, compare_type)

    # Handle report file selection
    elif data.startswith("report_"):
        try:
            await query.answer()  # Acknowledge callback query
            index_str = data[7:]  # remove "report_"
            session = get_user_session(query.from_user.id)
            if not hasattr(session, "selected_reports") or not session.selected_reports:
                await query.edit_message_text("❌ No report list found in session.")
                return
            try:
                index = int(index_str)
            except ValueError:
                await query.edit_message_text("❌ Invalid selection.")
                return
            if index < 0 or index >= len(session.selected_reports):
                await query.edit_message_text("❌ Selection out of range.")
                return
            file_name = session.selected_reports[index]
            folder_path = r"D:\TelegramBot\Last month reports"
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    await query.message.reply_document(document=f, filename=file_name)
            else:
                await query.edit_message_text("❌ File not found.")
        except Exception as e:
            logger.error(f"Error sending report file '{file_name}': {e}")
            await query.edit_message_text(f"❌ Error sending file: {str(e)}")

# Additional helper functions
async def compare_service_operators(query, service):
    """Compare a service across all operators"""
    service_norm, service_full = detect_service(service)
    if not service_norm:
        await query.edit_message_text("❌ Service not found.")
        return
    
    q = df.copy()
    op_data = {}
    for op, aliases in OP_GROUPS.items():
        op_df = q[q["_op_norm"].isin(aliases)]
        service_rows = get_service_data(op_df, service_norm)
        total = service_rows["YTD Actual"].sum()
        if total > 0:
            op_data[op.capitalize()] = total
    
    if not op_data:
        await query.edit_message_text(f"❌ No data for {service_full}.")
        return
    
    lines = [f"🆚 <b>{service_full} - Comparison</b>"]
    for op, val in sorted(op_data.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"• {op}: {format_num(val)}")
    
    await query.edit_message_text("\n".join(lines), parse_mode="HTML")

async def compare_two_operators(query, op1, op2):
    """Compare two operators"""
    q = df.copy()
    
    # Get data for operator 1
    op1_aliases = OP_GROUPS.get(op1, [])
    op1_df = q[q["_op_norm"].isin(op1_aliases)]
    op1_total = op1_df["YTD Actual"].sum()
    
    # Get data for operator 2
    op2_aliases = OP_GROUPS.get(op2, [])
    op2_df = q[q["_op_norm"].isin(op2_aliases)]
    op2_total = op2_df["YTD Actual"].sum()
    
    winner = "🟢" if op1_total > op2_total else "🔵" if op2_total > op1_total else "🟰"
    
    reply = (
        f"🆚 <b>Compare: {op1.upper()} vs {op2.upper()}</b>\n"
        f"🟢 {op1.capitalize()}: {format_num(op1_total)}\n"
        f"🔵 {op2.capitalize()}: {format_num(op2_total)}\n"
        f"{winner} <b>Winner</b>"
    )
    
    await query.edit_message_text(reply, parse_mode="HTML")

# =========================
# Updated Handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تحديث دالة البداية مع الكيبورد"""
    welcome_msg = (
        "🎉 <b>مرحبًا بك في بوت تحليل الإيرادات!</b>\n\n"
        "📊 يمكنني مساعدتك في:\n"
        "• تحليل إيرادات الشركات\n"
        "• مقارنة الأداء\n"
        "• عرض الاتجاهات السنوية والشهرية\n"
        "• إنشاء التقارير\n\n"
        "👇 اختر من القائمة أدناه أو اكتب سؤالك مباشرة:"
    )

    await update.message.reply_text(
        welcome_msg,
        parse_mode="HTML",
        reply_markup=get_main_keyboard()
    )

async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """التعامل مع اختيارات القائمة الرئيسية"""
    text = update.message.text
    user_id = update.effective_user.id
    session = get_user_session(user_id)

    if text == "📊 Total Revenues":
        session.current_action = "total_revenues"
        await update.message.reply_text(
            "🏢 اختر المشغل:",
            reply_markup=get_operators_keyboard()
        )

    elif text == "📈 Service Trends":
        session.current_action = "service_trends"
        await update.message.reply_text(
            "🔧 اختر الخدمة للاتجاه:",
            reply_markup=get_services_keyboard()
        )

    elif text == "📅 Monthly Trends":
        session.current_action = "monthly_trends_service"
        await update.message.reply_text(
            "🔧 اختر الخدمة للاتجاه الشهري:",
            reply_markup=get_services_keyboard()
        )

    elif text == "🆚 Compare Operators":
        session.current_action = "compare_operators"
        await update.message.reply_text(
            "📊 اختر نوع المقارنة:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("👥 All Operators", callback_data="compare_all")],
                [InlineKeyboardButton("🆚 Two Operators", callback_data="compare_two")],
                [InlineKeyboardButton("📋 By Service", callback_data="compare_service")],
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
            ])
        )

    elif text == "📋 Summary Report":
        session.current_action = "summary"
        await update.message.reply_text(
            "📅 اختر السنة:",
            reply_markup=get_years_keyboard()
        )

    elif text == "🔧 All Services":
        await services(update, context)

    elif text == "🤖 Smart AI":
        await update.message.reply_text(
            "🧠 اختر سؤال سريع أو اكتب سؤالك:",
            reply_markup=get_ai_questions_keyboard()
        )

    elif text == "📞 Help":
        await help_cmd(update, context)

    elif text == "🔄 Last Query":
        await last(update, context)

    elif text == "📄 Last Month Reports":
        await report(update, context)

    else:
        # Handle normal text queries
        await handle_text_query(update, context)

async def handle_text_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """التعامل مع الاستعلامات النصية العادية"""
    text = update.message.text or ""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    first_name = update.effective_user.first_name or "Unknown"

    # Log the query
    log_query(user_id, username, first_name, text)
    query_log.append(text.lower().strip())

    # Save last query
    user_last_query[user_id] = text

    logger.info(f"User {user_id} ({first_name}): {text}")

    # Check for report command
    if any(word in norm(text) for word in ["report", "تقرير"]):
        await report(update, context)
        return

    # Try normal bot logic first
    try:
        reply = compute_answer(text)

        if "❌ I couldn't detect" not in reply and "No data" not in reply:
            await update.message.reply_text(
                reply,
                parse_mode="HTML",
                reply_markup=get_main_keyboard()
            )
            return
    except Exception as e:
        logger.error(f"Error in compute_answer: {e}")

    # Use Gemini if normal logic fails
    if gemini_model:
        year = detect_year(text)
        op_canonical, op_match, _ = detect_operator(text)
        service_norm, _ = detect_service(text)

        if any(word in norm(text) for word in ["contribution", "percentage", "share", "نسبة", "موزعة", "توزيع"]):
            contrib = get_contribution_summary(year, op_canonical)
            if contrib:
                contributions, total = contrib
                context_data = f"Total Revenue: {format_num(total)}\n"
                for svc, data in contributions.items():
                    context_data += f"• {svc.title()}: {format_num(data['value'])} ({data['percent']:.1f}%)\n"
            else:
                context_data = "No revenue data found for the requested filters."
        else:
            context_data = "No specific calculation needed."

        lang = "العربية" if any('\u0600' <= c <= '\u06FF' for c in text) else "English"

        prompt = f"""
        You are a telecom revenue analyst.
        Question: "{text}"
        Context: Real revenue data from 2020 to 2025 for Vodafone, Etisalat, Orange, and Others.

        Calculated Data:
        {context_data}

        Answer in {lang}, clearly and professionally.
        """

        ai_reply = ask_gemini(prompt)
        await update.message.reply_text(
            ai_reply,
            reply_markup=get_main_keyboard()
        )
    else:
        await update.message.reply_text(
            "❌ ميزة الذكاء الاصطناعي غير متاحة.",
            reply_markup=get_main_keyboard()
        )

# =========================
# Telegram handlers
# =========================
HELP_TEXT = (
    "👋 Hi! I'm your Revenue Bot.\n\n"
    "<b>📊 Ask me:</b>\n"
    "• Total Revenues Vodafone 2025\n"
    "• tx etisalat until now\n"
    "• ADSL Orange\n\n"
    "<b>🔍 Smart AI:</b>\n"
    "• /ai What is the contribution of each service?\n"
    "• /ai How is revenue distributed?\n"
    "• /ai Why did revenue drop?\n\n"
    "<b>🔧 Commands:</b>\n"
    "• /trend transmission\n"
    "• /monthly_trend total revenues\n"
    "• /chart adsl\n"
    "• /compare_main 2025\n"
    "• /compare_tx_budget transmission 2025\n"
    "• /summary 2025\n"
    "• /services\n"
    "• /last"
)

HELP_TEXT_AR = (
    "👋 مرحبًا! أنا بوت الإيرادات.\n\n"
    "<b>📊 اسألني:</b>\n"
    "• إيرادات فودافون 2025\n"
    "• ترانسميشن اتصالات\n"
    "• إيه هي نسبة إنجاز اورنج؟\n\n"
    "<b>🔍 الذكاء الاصطناعي:</b>\n"
    "• /ai ما هي نسبة كل خدمة من الإيرادات؟\n"
    "• /ai كيف توزعت الإيرادات؟\n"
    "• /ai ليه الإيرادات نزلت؟\n\n"
    "<b>🔧 الأوامر:</b>\n"
    "• /trend ترانسميشن\n"
    "• /monthly_trend إجمالي الإيرادات\n"
    "• /chart ادسل\n"
    "• /compare_main 2025\n"
    "• /compare_tx_budget ترانسميشن 2025\n"
    "• /summary 2025\n"
    "• /services\n"
    "• /last"
)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text if update.message else ""
    is_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
    msg = HELP_TEXT_AR if is_arabic else HELP_TEXT
    await update.message.reply_text(msg, parse_mode="HTML")

async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Use: /trend <service>\nExample: /trend transmission", parse_mode="HTML")
        return

    service_text = " ".join(context.args)
    service_norm, service_full = detect_service(service_text)

    if not service_norm:
        await update.message.reply_text("❌ Service not found.", parse_mode="HTML")
        return

    q = df.copy()
    
    # Special handling for total revenues
    if service_norm == "total revenues":
        yearly = {}
        for year in q["Year"].unique():
            if year and year > 0:
                year_df = q[q["Year"] == year]
                total = 0
                for svc in ["infrastructure", "access", "voice"]:
                    service_data = get_service_data(year_df, svc)
                    total += service_data["YTD Actual"].sum()
                yearly[year] = total
    else:
        yearly = q.groupby("Year").apply(lambda g: get_service_data(g, service_norm)["YTD Actual"].sum())
        yearly = yearly[yearly.index > 0]

    if yearly.empty if isinstance(yearly, pd.Series) else not yearly:
        await update.message.reply_text("❌ No data for this service over years.")
        return

    # Convert to dict if Series
    if isinstance(yearly, pd.Series):
        yearly = yearly.to_dict()
    
    years = sorted(yearly.keys())
    values = [yearly[y] for y in years]

    plt.figure(figsize=(10, 5))
    plt.plot(years, values, marker='o', color='blue', linewidth=2, markersize=6)
    for year, value in zip(years, values):
        plt.text(year, value + max(values)*0.02, format_num(value), ha='center', fontsize=9, color='blue')
    plt.title(f"Trend: {service_full}", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Revenue")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    await update.message.reply_photo(
        photo=buf,
        caption=f"📈 Trend for <b>{service_full}</b>",
        parse_mode="HTML"
    )

async def monthly_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command for monthly trend"""
    if not context.args:
        await update.message.reply_text(
            "❌ Use: /monthly_trend <service>\nExample: /monthly_trend total revenues", 
            parse_mode="HTML"
        )
        return

    service_text = " ".join(context.args)
    service_norm, service_full = detect_service(service_text)

    if not service_norm:
        await update.message.reply_text("❌ Service not found.", parse_mode="HTML")
        return

    # Check if month data exists
    if "Year_Month" not in df.columns:
        await update.message.reply_text("❌ Monthly data not available in the dataset.")
        return

    q = df[df["Year_Month"].notna()].copy()
    
    # Special handling for total revenues
    if service_norm == "total revenues":
        monthly_data = {}
        for ym in q["Year_Month"].unique():
            month_df = q[q["Year_Month"] == ym]
            total = 0
            for svc in ["infrastructure", "access", "voice"]:
                service_data = get_service_data(month_df, svc)
                total += service_data["YTD Actual"].sum()
            if total > 0:
                monthly_data[ym] = total
    else:
        monthly_data = {}
        for ym in q["Year_Month"].unique():
            month_df = q[q["Year_Month"] == ym]
            service_data = get_service_data(month_df, service_norm)
            value = service_data["YTD Actual"].sum()
            if value > 0:
                monthly_data[ym] = value
    
    if not monthly_data:
        await update.message.reply_text("❌ No monthly data for this service.")
        return

    # Sort by year-month
    months = sorted(monthly_data.keys())
    values = [monthly_data[m] for m in months]
    
    # Create chart
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(months)), values, marker='o', color='green', linewidth=2, markersize=5)
    
    # Add value labels (skip some for clarity if too many)
    step = max(1, len(months) // 12)
    for i in range(0, len(months), step):
        plt.text(i, values[i] + max(values)*0.02, format_num(values[i]), 
                ha='center', fontsize=8, color='green')
    
    plt.title(f"Monthly Trend: {service_full}", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(range(0, len(months), max(1, len(months)//12)), 
               [months[i] for i in range(0, len(months), max(1, len(months)//12))], 
               rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    await update.message.reply_photo(
        photo=buf,
        caption=f"📅 Monthly Trend for <b>{service_full}</b>",
        parse_mode="HTML"
    )

async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text("❌ Use: /chart <service>\nExample: /chart transmission", parse_mode="HTML")
        return

    year = detect_year(text)
    op_canonical, op_match, _ = detect_operator(text)
    service_norm, service_full = detect_service(text)

    if not service_norm:
        await update.message.reply_text("❌ Service not detected.", parse_mode="HTML")
        return

    q = df.copy()
    if year and "Year" in q.columns:
        q = q[q["Year"] == year]
    if op_canonical:
        q = q[q["_op_norm"].isin(OP_GROUPS[op_canonical])]

    service_rows = get_service_data(q, service_norm)
    actual = service_rows["YTD Actual"].sum()
    budget = service_rows["YTD Budget"].sum()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["YTD Actual", "YTD Budget"], [actual, budget], color=["#AAACAB", "#9E049E"])
    plt.title(f"{service_full} | {year or 'All Years'}", fontsize=12)
    plt.ylabel("Revenue")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(actual, budget)*0.02,
                 format_num(height), ha='center', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    await update.message.reply_photo(photo=buf, caption="📊 Actual vs Budget")

async def compare_main(update: Update, context: ContextTypes.DEFAULT_TYPE):
    year = int(context.args[0]) if context.args and context.args[0].isdigit() else None
    q = df.copy()
    if year:
        q = q[q["Year"] == year]

    total_infra = get_service_data(q, "infrastructure")["YTD Actual"].sum()
    total_access = get_service_data(q, "access")["YTD Actual"].sum()
    total_voice = get_service_data(q, "voice")["YTD Actual"].sum()

    if all(v == 0 for v in [total_infra, total_access, total_voice]):
        await update.message.reply_text("❌ No data found for main services.")
        return

    ytxt = f" {year}" if year else " Overall"

    reply = (
        f"📊 <b>Main Services Comparison{ytxt}</b>\n"
        f"🔹 <b>Infrastructure:</b> {format_num(total_infra)}\n"
        f"🔸 <b>Access:</b> {format_num(total_access)}\n"
        f"🔺 <b>Voice:</b> {format_num(total_voice)}"
    )
    await update.message.reply_text(reply, parse_mode="HTML")

async def compare_tx_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "❌ Use: /compare_tx_budget <service> [year]\n"
            "Example: /compare_tx_budget transmission 2025",
            parse_mode="HTML"
        )
        return

    service_text = context.args[0]
    year = int(context.args[1]) if len(context.args) > 1 and context.args[1].isdigit() else None

    service_norm, service_full = detect_service(service_text)
    if not service_norm:
        await update.message.reply_text("❌ Service not found.", parse_mode="HTML")
        return

    q = df.copy()
    if year:
        q = q[q["Year"] == year]

    op_data = {}
    for op, aliases in OP_GROUPS.items():
        op_aliases = set(alias for alias in aliases)
        op_df = q[q["_op_norm"].apply(lambda x: any(alias in x for alias in op_aliases))]
        service_rows = get_service_data(op_df, service_norm)
        actual = service_rows["YTD Actual"].sum()
        budget = service_rows["YTD Budget"].sum()
        if actual > 0 or budget > 0:
            op_data[op.capitalize()] = {"actual": actual, "budget": budget}

    if not op_data:
        await update.message.reply_text("❌ No data for this service and operators.")
        return

    operators = list(op_data.keys())
    actuals = [op_data[op]["actual"] for op in operators]
    budgets = [op_data[op]["budget"] for op in operators]

    x = range(len(operators))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar([p - width/2 for p in x], actuals, width, label='YTD Actual', color='#AAACAB')
    bars2 = plt.bar([p + width/2 for p in x], budgets, width, label='YTD Budget', color='#9E049E')
    plt.xlabel('Operator')
    plt.ylabel('Revenue')
    plt.title(f"Actual vs Budget: {service_full} {'| ' + str(year) if year else ''}")
    plt.xticks(x, operators)
    plt.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height + max(max(actuals), max(budgets))*0.02,
                format_num(height), ha='center', fontsize=9
            )
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    await update.message.reply_photo(
        photo=buf,
        caption=f"📊 Actual vs Budget for <b>{service_full}</b> per Operator",
        parse_mode="HTML"
    )

async def trends(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args) if context.args else ""
    op_canonical, op_match, _ = detect_operator(text)
    if not op_canonical:
        await update.message.reply_text("❌ Please specify an operator. Example: /trends vodafone", parse_mode="HTML")
        return

    q = df.copy()
    if op_canonical:
        q = q[q["_op_norm"].isin(OP_GROUPS[op_canonical])]

    yearly = q.groupby("Year")["YTD Actual"].sum().sort_index()
    lines = [f"📈 <b>{op_match.capitalize()} - Revenue Trends</b>"]
    for year, val in yearly.items():
        if year > 0:
            lines.append(f"• {year}: {format_num(val)}")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")

async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("❌ Use: /compare op1 op2 [year]")
        return
    op1, op2 = args[0], args[1]
    year = int(args[2]) if len(args) > 2 and args[2].isdigit() else None
    def get_val(op_text, yr):
        op_canon, _, _ = detect_operator(op_text)
        if not op_canon: return 0
        q = df.copy()
        if yr: q = q[q["Year"] == yr]
        q = q[q["_op_norm"].isin(OP_GROUPS[op_canon])]
        return q["YTD Actual"].sum()

    v1, v2 = get_val(op1, year), get_val(op2, year)
    ytxt = f" {year}" if year else " Latest"
    winner = "🟢" if v1 > v2 else "🔵" if v2 > v1 else "🟰"
    reply = (
        f"🆚 <b>Compare: {op1.upper()} vs {op2.upper()}{ytxt}</b>\n"
        f"🟢 {op1.capitalize()}: {format_num(v1)}\n"
        f"🔵 {op2.capitalize()}: {format_num(v2)}\n"
        f"{winner} <b>Winner</b>"
    )
    await update.message.reply_text(reply, parse_mode="HTML")

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    year = int(context.args[0]) if context.args and context.args[0].isdigit() else None
    q = df.copy()
    if year:
        q = q[q["Year"] == year]

    data = {}
    for op, aliases in OP_GROUPS.items():
        op_aliases = set(alias for alias in aliases)
        op_df = q[q["_op_norm"].apply(lambda x: any(alias in x for alias in op_aliases))]
        total = op_df["YTD Actual"].sum()
        if total > 0:
            data[op] = total

    if not data:
        await update.message.reply_text("❌ No data found for any operator.")
        return

    lines = [f"📋 <b>Summary{' ' + str(year) if year else ''}</b>"]
    for op, val in sorted(data.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"• {op.capitalize()}: {format_num(val)}")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")

async def services(update: Update, context: ContextTypes.DEFAULT_TYPE):
    servs = sorted(FLAT_SERVICES.keys())
    msg = "🔧 <b>Supported Services:</b>\n" + "\n".join(f"• {s}" for s in servs)
    await update.message.reply_text(msg, parse_mode="HTML")

async def last(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    last_text = user_last_query.get(user_id)
    if not last_text:
        await update.message.reply_text("❌ No previous query.")
        return
    reply = compute_answer(last_text)
    await update.message.reply_text(reply, parse_mode="HTML")

async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Use the specified path for the reports folder
        folder_path = r"D:\TelegramBot\Last month reports"
        logger.info(f"Checking folder: {folder_path}")

        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            await update.message.reply_text("❌ Folder 'Last month reports' not found in the current working directory. Please create the folder and add PDF files.")
            return

        # الحصول على جميع الملفات في المجلد
        all_files = os.listdir(folder_path)
        logger.info(f"Files in folder: {all_files}")

        # تصفية الملفات PDF فقط
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
        logger.info(f"PDF files found: {pdf_files}")

        if not pdf_files:
            await update.message.reply_text("❌ No PDF files found in 'Last month reports' folder.")
            return

        # ترتيب الملفات أبجدياً مع مراعاة الأرقام في العناوين
        import re

        def extract_number(filename):
            match = re.match(r"(\d+)", filename)
            return int(match.group(1)) if match else float('inf')

        pdf_files.sort(key=extract_number)

        # Store in session for callback
        session = get_user_session(update.effective_user.id)
        session.selected_reports = pdf_files

        # إنشاء لوحة المفاتيح
        keyboard = []
        for i, pdf in enumerate(pdf_files):
            keyboard.append([InlineKeyboardButton(pdf, callback_data=f"report_{i}")])
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="cancel")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("📄 Select a report to download:", reply_markup=reply_markup)

    except Exception as e:
        logger.error(f"Error in report command: {e}")
        await update.message.reply_text(f"❌ An error occurred while processing the report command: {str(e)}")

# --- AI Command ---
async def ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args) if context.args else user_last_query.get(update.effective_user.id)
    if not text:
        await update.message.reply_text("❌ اكتب سؤالك أو استخدم /ai بعد سؤال.")
        return

    if gemini_model:
        year = detect_year(text)
        op_canonical, op_match, _ = detect_operator(text)

        if any(word in norm(text) for word in ["contribution", "percentage", "share", "نسبة", "موزعة", "توزيع"]):
            contrib = get_contribution_summary(year, op_canonical)
            if contrib:
                contributions, total = contrib
                context_data = f"Total Revenue: {format_num(total)}\n"
                for svc, data in contributions.items():
                    context_data += f"• {svc.title()}: {format_num(data['value'])} ({data['percent']:.1f}%)\n"
            else:
                context_data = "No revenue data found for the requested filters."
        else:
            context_data = "No specific calculation needed."

        lang = "العربية" if any('\u0600' <= c <= '\u06FF' for c in text) else "English"

        additional_context = """
        The bot can answer:
        - Revenue queries (YTD, Budget, VAR)
        - Trends over years
        - Comparisons between operators/services
        - Service hierarchy (e.g., tx iru → transmission → infrastructure)
        - Progress vs budget
        - Market share estimates
        """

        prompt = f"""
        You are a telecom revenue analyst.
        Question: "{text}"
        Context: The user has access to real revenue data from 2020 to 2025 for Vodafone, Etisalat, Orange, and Others.
        Available services: Infrastructure, Access, Voice.

        Calculated Data Summary:
        {context_data}

        {additional_context}

        Instructions:
        - Answer in {lang}, clearly and professionally.
        - Use the provided numbers directly.
        - If percentages are available, mention them.
        - If the question is about trends, explain logically.
        - Never say 'I don't have data' if data is provided above.
        - Be concise and helpful.
        """

        reply = ask_gemini(prompt)
        await update.message.reply_text(reply)
    else:
        await update.message.reply_text("❌ ميزة الذكاء الاصطناعي غير متاحة.")

# --- Statistics Command (for admin only) ---
ADMIN_USER_ID = 1627582594  # ← غيره لرقمك

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Access denied.")
        return

    if not query_log:
        await update.message.reply_text("📭 No queries logged yet.")
        return

    counter = Counter(query_log)
    top_queries = counter.most_common(10)

    lines = ["📊 <b>Top 10 User Questions</b>\n"]
    for i, (q, count) in enumerate(top_queries, 1):
        lines.append(f"{i}. <code>{q}</code> → <b>{count}</b> time(s)")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")

# =========================
# Run bot
# =========================
def main():
    app = Application.builder().token(TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("trends", trends))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("services", services))
    app.add_handler(CommandHandler("last", last))
    app.add_handler(CommandHandler("trend", trend))
    app.add_handler(CommandHandler("chart", chart))
    app.add_handler(CommandHandler("monthly_trend", monthly_trend))
    app.add_handler(CommandHandler("compare_main", compare_main))
    app.add_handler(CommandHandler("compare_tx_budget", compare_tx_budget))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("ai", ai))
    app.add_handler(CommandHandler("report", report))

    # Callback query handler
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    # Message handlers
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_main_menu
    ))

    logger.info("🚀 Revenue Bot Plus Smart is running... (With Fixed Interactive Keyboards)")
    app.run_polling()

if __name__ == "__main__":
    main()