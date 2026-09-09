import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from ai_engine import (
    analyze_with_gemini,
    fetch_idx_stock_data,
    fetch_us_stock_data,
)

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Investment Assistant", 
    layout="wide"
)

# Header Utama
st.title("Investment Assistant")
st.caption("Analisis Saham US & Indonesia Berbasis AI & Data Pasar Real-Time")

# Sidebar - Form Input Parameter
st.sidebar.header("Pengaturan Analisis")
market = st.sidebar.selectbox(
    "Pilih Pasar:",
    ["Saham US", "Saham Indonesia"],
)

default_ticker = "NVDA" if "US" in market else "BBCA"
ticker_input = st.sidebar.text_input("Kode Ticker Saham:", value=default_ticker)

risk_profile = st.sidebar.select_slider(
    "Profil Risiko Kamu:",
    options=["Konservatif", "Moderat", "Agresif"],
    value="Moderat",
)

btn_analyze = st.sidebar.button(
    "Jalankan Analisis", use_container_width=True
)

# Area Utama Dashboard
col_data, col_chart = st.columns([1, 1])

if btn_analyze:
    with st.spinner("Mengambil data pasar & menganalisis dengan Gemini AI"):
        # 1. Fetch Data Sesuai Pasar
        if "US" in market:
            stock_data = fetch_us_stock_data(ticker_input)
            yf_ticker = ticker_input.upper().strip()
        else:
            clean_ticker = ticker_input.upper().replace(".JK", "").strip()
            stock_data = fetch_idx_stock_data(clean_ticker)
            yf_ticker = f"{clean_ticker}.JK"

        if not stock_data:
            st.error(
                f"Gagal mengambil data untuk ticker '{ticker_input}'. Periksa kembali kodenya."
            )
        else:
            # 2. Tampilkan Ringkasan Metrik Data Secara Rapi
            with col_data:
                st.subheader(
                    f"Metrik {stock_data.get('Nama Perusahaan', ticker_input)}"
                )
                
                # Tampilkan Ringkasan Metrik Utama
                m_col1, m_col2 = st.columns(2)
                price_key = "Harga Terakhir ($)" if "US" in market else "Harga Terakhir (Rp)"
                curr_symbol = "$" if "US" in market else "Rp "
                
                with m_col1:
                    price_val = stock_data.get(price_key, 'N/A')
                    st.metric(
                        label="Harga Terakhir", 
                        value=f"{curr_symbol}{price_val:,}" if isinstance(price_val, (int, float)) else str(price_val),
                        delta=f"{stock_data.get('Perubahan (%)', 0)}%" if "Perubahan (%)" in stock_data else None
                    )
                with m_col2:
                    st.metric(label="P/E Ratio", value=str(stock_data.get("PE Ratio", "N/A")))

                # Tampilkan detail metrik lainnya dalam kontainer
                st.json(stock_data, expanded=False)

            # 3. Tampilkan Grafik Candlestick
            with col_chart:
                st.subheader("Grafik Harga (6 Bulan Terakhir)")
                try:
                    df_hist = yf.Ticker(yf_ticker).history(period="6mo")
                    if not df_hist.empty:
                        fig = go.Figure(
                            data=[
                                go.Candlestick(
                                    x=df_hist.index,
                                    open=df_hist["Open"],
                                    high=df_hist["High"],
                                    low=df_hist["Low"],
                                    close=df_hist["Close"],
                                )
                            ]
                        )
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=350,
                            xaxis_rangeslider_visible=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Data grafik tidak tersedia untuk ticker ini.")
                except Exception as e:
                    st.warning(f"Gagal memuat grafik: {e}")

            # 4. Tampilkan Hasil Analisis Gemini AI dengan Penanganan Error
            st.markdown("---")
            st.subheader("Rekomendasi & Strategi Investasi AI")
            
            try:
                ai_result = analyze_with_gemini(stock_data, risk_profile)
                st.info(ai_result)
            except Exception as e:
                st.error(f"Gagal mendapatkan analisis dari Gemini AI: {e}")

else:
    st.info(
        "Masukkan kode saham di sidebar kiri dan klik **Jalankan Analisis** untuk mulai."
    )