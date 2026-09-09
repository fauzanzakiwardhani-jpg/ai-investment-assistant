import os
import time
import requests
import streamlit as st
import yfinance as yf
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from dotenv import load_dotenv

# 1. Load API Keys (Lokal .env)
load_dotenv()

# 2. Ambil GEMINI_API_KEY (Prioritas: Streamlit Cloud Secrets -> .env Lokal)
GEMINI_API_KEY = None
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Ambil FMP_API_KEY
FMP_API_KEY = None
if "FMP_API_KEY" in st.secrets:
    FMP_API_KEY = st.secrets["FMP_API_KEY"]
else:
    FMP_API_KEY = os.getenv("FMP_API_KEY")

# Validasi API Key sebelum inisialisasi Client
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY tidak ditemukan! Pastikan telah mengaturnya di Secrets Streamlit Cloud atau file .env lokal."
    )

# Inisialisasi Client Google Gemini AI
client = genai.Client(api_key=GEMINI_API_KEY)


def fetch_us_stock_data(ticker: str):
    """Menarik data Saham US dari FMP API dengan Fallback ke yfinance"""
    symbol = ticker.upper().strip()
    
    # 1. Coba tarik data via FMP API
    if FMP_API_KEY:
        try:
            url_profile = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FMP_API_KEY}"
            url_quote = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"

            prof_res = requests.get(url_profile, timeout=5).json()
            quote_res = requests.get(url_quote, timeout=5).json()

            if prof_res and quote_res and isinstance(prof_res, list) and isinstance(quote_res, list):
                prof = prof_res[0]
                q = quote_res[0]

                return {
                    "Pasar": "US",
                    "Ticker": symbol,
                    "Nama Perusahaan": prof.get("companyName", symbol),
                    "Sektor": prof.get("sector", "N/A"),
                    "Harga Terakhir ($)": q.get("price", "N/A"),
                    "Perubahan (%)": round(q.get("changesPercentage", 0), 2),
                    "PE Ratio": round(q.get("pe", 0), 2) if q.get("pe") else "N/A",
                    "Market Cap ($)": prof.get("mktCap", "N/A"),
                    "Deskripsi Singkat": (
                        prof.get("description", "")[:250] + "..."
                        if prof.get("description")
                        else "Tidak ada deskripsi."
                    ),
                    "Sumber Data": "Financial Modeling Prep"
                }
        except Exception as e:
            print(f"FMP API Error: {e}")

    # 2. Fallback ke yfinance
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        if not info:
            return None

        current_price = (
            info.get("currentPrice") 
            or info.get("regularMarketPrice") 
            or info.get("previousClose")
        )
        if not current_price:
            return None

        return {
            "Pasar": "US",
            "Ticker": symbol,
            "Nama Perusahaan": info.get("longName", symbol),
            "Sektor": info.get("sector", "N/A"),
            "Harga Terakhir ($)": current_price,
            "Perubahan (%)": round(info.get("regularMarketChangePercent", 0), 2),
            "PE Ratio": (
                round(info.get("trailingPE", 0), 2)
                if isinstance(info.get("trailingPE"), (int, float))
                else "N/A"
            ),
            "Market Cap ($)": info.get("marketCap", "N/A"),
            "Deskripsi Singkat": (
                info.get("longBusinessSummary", "")[:250] + "..."
                if info.get("longBusinessSummary")
                else "Tidak ada deskripsi."
            ),
            "Sumber Data": "Yahoo Finance (Fallback)"
        }
    except Exception as e:
        print(f"Error yfinance US Stock: {e}")
        return None


def fetch_idx_stock_data(ticker: str):
    """Menarik data Saham Indonesia dari yfinance dengan fleksibilitas harga & metrik"""
    clean_symbol = ticker.upper().replace(".JK", "").strip()
    full_symbol = f"{clean_symbol}.JK"

    try:
        stock = yf.Ticker(full_symbol)
        info = stock.info

        if not info:
            return None

        # Fallback bertahap untuk mengambil harga terbaru
        current_price = (
            info.get("currentPrice") 
            or info.get("regularMarketPrice") 
            or info.get("previousClose")
            or info.get("open")
        )

        if not current_price:
            return None

        # Pengambilan P/E Ratio dengan pembacaan aman
        pe_val = info.get("trailingPE") or info.get("forwardPE")
        pe_ratio = round(pe_val, 2) if isinstance(pe_val, (int, float)) else "N/A"

        # Pengambilan PBV Ratio dengan pembacaan aman
        pbv_val = info.get("priceToBook")
        pbv_ratio = round(pbv_val, 2) if isinstance(pbv_val, (int, float)) else "N/A"

        # Pengambilan Dividend Yield dengan pembacaan aman
        div_val = info.get("dividendYield")
        div_yield = round(div_val * 100, 2) if isinstance(div_val, (int, float)) else 0.0

        return {
            "Pasar": "Indonesia",
            "Ticker": clean_symbol,
            "Nama Perusahaan": info.get("longName") or info.get("shortName") or clean_symbol,
            "Sektor": info.get("sector", "N/A"),
            "Harga Terakhir (Rp)": current_price,
            "PE Ratio": pe_ratio,
            "PBV Ratio": pbv_ratio,
            "Dividend Yield (%)": div_yield,
            "Market Cap (Rp)": info.get("marketCap", "N/A"),
            "Sumber Data": "Yahoo Finance (Indonesia)"
        }
    except Exception as e:
        print(f"Error yfinance IDX Stock ({clean_symbol}): {e}")
        return None


def analyze_with_gemini(stock_data: dict, risk_profile: str):
    """Analisis otomatis via Gemini API dengan daftar model resmi dan retry mechanism"""
    system_instruction = """
    Kamu adalah Wall Street & BEI Senior Investment Analyst berpengalaman.
    Tugasmu adalah menganalisis data saham terstruktur dari API dan memberikan laporan rekomendasi investasi yang profesional.

    Format Laporan:
    1. 📌 RINGKASAN & BISNIS EMITEN (1-2 kalimat singkat)
    2. 📊 ANALISIS VALUASI & METRIK (P/E, P/B, Dividen, atau Growth)
    3. ⚖️ FIT PLATFORM & RISIKO:
       - Saham US: Hubungkan dengan konteks makro US & dampak kurs USD/IDR.
       - Saham Indonesia: Hubungkan dengan ketahanan dividen & stabilitas IHSG.
    4. 💡 STRATEGI EKSEKUSI (Beli / Wait & See / Sell), Rekomendasi Dollar-Cost Averaging (DCA), serta Target Risiko.
    
    Gunakan bahasa Indonesia yang lugas, terstruktur, dan profesional. Selalu sertakan disclaimer risiko di akhir.
    """

    prompt = f"Profil Risiko Investor: {risk_profile}\nData Saham dari API:\n{stock_data}"

    # Urutan model yang dicoba bertahap
    models_to_try = ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3,
    )

    last_exception = None

    for model_name in models_to_try:
        # Retry hingga 3x jika server sibuk (503)
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text
            except ServerError:
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Wait 1s, then 2s
                    continue
                break
            except APIError as e:
                last_exception = e
                # Jika model 404/not found, langsung lompat ke model berikutnya
                break

    raise RuntimeError(f"Gagal memanggil Gemini API: {last_exception}")
