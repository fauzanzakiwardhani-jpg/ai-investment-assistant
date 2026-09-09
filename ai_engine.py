import os
import time
import requests
import yfinance as yf
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from dotenv import load_dotenv

# Load API Keys dari file .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

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

        if not info or ("regularMarketPrice" not in info and "currentPrice" not in info):
            return None

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        return {
            "Pasar": "US",
            "Ticker": symbol,
            "Nama Perusahaan": info.get("longName", symbol),
            "Sektor": info.get("sector", "N/A"),
            "Harga Terakhir ($)": current_price,
            "Perubahan (%)": round(info.get("regularMarketChangePercent", 0), 2),
            "PE Ratio": (
                round(info.get("trailingPE", 0), 2)
                if info.get("trailingPE")
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
    """Menarik data Saham Indonesia dari yfinance"""
    clean_symbol = ticker.upper().replace(".JK", "").strip()
    full_symbol = f"{clean_symbol}.JK"

    try:
        stock = yf.Ticker(full_symbol)
        info = stock.info

        if not info or (
            "regularMarketPrice" not in info and "currentPrice" not in info
        ):
            return None

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        return {
            "Pasar": "Indonesia",
            "Ticker": clean_symbol,
            "Nama Perusahaan": info.get("longName", clean_symbol),
            "Sektor": info.get("sector", "N/A"),
            "Harga Terakhir (Rp)": current_price,
            "PE Ratio": (
                round(info.get("trailingPE", 0), 2)
                if info.get("trailingPE")
                else "N/A"
            ),
            "PBV Ratio": (
                round(info.get("priceToBook", 0), 2)
                if info.get("priceToBook")
                else "N/A"
            ),
            "Dividend Yield (%)": (
                round(info.get("dividendYield", 0) * 100, 2)
                if info.get("dividendYield")
                else 0
            ),
            "Market Cap (Rp)": info.get("marketCap", "N/A"),
            "Sumber Data": "Yahoo Finance (Indonesia)"
        }
    except Exception as e:
        print(f"Error yfinance IDX Stock: {e}")
        return None


def analyze_with_gemini(stock_data: dict, risk_profile: str):
    """Analisis otomatis via Gemini API dengan daftar model resmi dan retry mechanism"""
    system_instruction = """
    Kamu adalah Wall Street & BEI Senior Investment Analyst berpengalaman.
    Tugasmu adalah menganalisis data saham terstruktur dari API dan memberikan laporan rekomendasi investasi yang profesional.

    Format Laporan:
    1. RINGKASAN & BISNIS EMITEN (1-2 kalimat singkat)
    2. ANALISIS VALUASI & METRIK (P/E, P/B, Dividen, atau Growth)
    3. FIT PLATFORM & RISIKO:
       - Saham US : Hubungkan dengan konteks makro US & dampak kurs USD/IDR.
       - Saham Indonesia : Hubungkan dengan ketahanan dividen & stabilitas IHSG.
    4. STRATEGI EKSEKUSI (Beli / Wait & See / Sell), Rekomendasi Dollar-Cost Averaging (DCA), serta Target Risiko.
    
    Gunakan bahasa Indonesia yang lugas, terstruktur, dan profesional. Selalu sertakan disclaimer risiko di akhir.
    """

    prompt = f"Profil Risiko Investor: {risk_profile}\nData Saham dari API:\n{stock_data}"

    # Urutan model resmi yang dicoba bertahap jika ada model yang error/busy
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
                # Jika model 404/not found, langsung lompat ke model berikutnya di list
                break

    raise RuntimeError(f"Gagal memanggil Gemini API: {last_exception}")