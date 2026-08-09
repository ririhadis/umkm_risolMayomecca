import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import time
from datetime import datetime
import holidays
import os
import pytz
import textwrap

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse
from google.oauth2.service_account import Credentials
import gspread
from telegram import Bot

#Konfigurasi dan Page Setup
st.set_page_config(
    page_title="SmartStock AI - Risol Mayo Mecca",
    page_icon="🥖",
    layout="wide"
)

#link dataset, model dan token
SPREADSHEET_ID = "1c7sG94xHTxR98rTUIPkyKkOaSy7c-je8edwoHtFvWJg"
gsheet_url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet=csv_penjualan"
gsheet_resep = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet=stok_bahan"
gholiday_url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet=hari_libur_custom"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "tft_model_produksi.pt")
telegram_token = st.secrets.get("TELEGRAM_TOKEN", "")
chat_id = st.secrets.get("CHAT_ID", "")

#safety buffer 5%
SAFETY_BUFFER = 1.05

#Memasukkan data terbaru ke google sheet
def get_gsheet_client():
    """Membuat koneksi ke Google Sheeets via Service Account (Streamlit Secrets)"""
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    return gspread.authorize(creds)

def append_data_to_gsheet(spreadsheets_id, new_row_data):
    """Menambahkan 1 baris data penjualan baru ke Google Sheets
    Membersihkan tipe data tipe numpy/datetime sebelum dikirim
    """
    try:
        client = get_gsheet_client()
        sheet = client.open_by_key(spreadsheets_id).worksheet("csv_penjualan")

        formatted_row = []
        for item in new_row_data:
            if isinstance(item, (pd.Timestamp, pd.DatetimeIndex)):
                formatted_row.append(item.strftime("%Y-%m-%d"))
            # PERBAIKAN: Tambahkan (int, float) Python standar
            elif isinstance(item, (int, float, np.integer, np.floating)):
                # Kirim sebagai angka asli (bukan string)
                formatted_row.append(
                    item.item() if hasattr(item, "item") else item
                )
            else:
                formatted_row.append(str(item) if item is not None else "")

        # PERBAIKAN: Gunakan USER_ENTERED agar Google Sheets mengenali tipe angka/tanggal otomatis
        sheet.append_row(formatted_row, value_input_option="USER_ENTERED")

        # Hapus cache Streamlit
        load_sales.clear()
        return True

    except Exception as e:
        st.error(f"Gagal menambahkan data ke Google Sheets: {e}")
        return False

#lOAD data & Model (Cached)
@st.cache_data(ttl=60)
def load_sales():
    #Mengambil historis data penjualan
    df = pd.read_csv(gsheet_url, parse_dates=["Tanggal"])
    df = df.set_index("Tanggal").sort_index()
    return df

def prepare_darts_data(df_input, target_cols, past_cov_cols):
    """Helper function untuk konversi DataFrame ke TimeSeries & Scaler Darts"""
    df_clean = df_input.copy()

    # Memastikan Index berupa DatetimeIndex murni & di-normalize
    if "Tanggal" in df_clean.columns:
        df_clean["Tanggal"] = pd.to_datetime(df_clean["Tanggal"])
        df_clean = df_clean.set_index("Tanggal")
    else:
        df_clean.index = pd.to_datetime(df_clean.index)

    df_clean.index = df_clean.index.normalize()
    df_clean = df_clean[~df_clean.index.duplicated(keep='last')].sort_index()

    # Memastikan kolom numerik
    all_needed_cols = list(target_cols) + list(past_cov_cols)
    for col in all_needed_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    #Resample harian
    #Membuat deret tanggal lengkap tanpa loncatan
    full_idx = pd.date_range(start=df_clean.index.min(), end=df_clean.index.max(), freq="D")
    df_clean = df_clean[all_needed_cols].reindex(full_idx).fillna(0)

    start_date = df_clean.index.min()
    
    # agar memenuhi kebutuhan output_chunk_length dari TFT Model
    max_sales_date = df_clean.index.max()
    extended_end_date = max_sales_date + pd.Timedelta(days=180)

    #Membuat TimeSeries Darts & Past Covariates
    y_ts = TimeSeries.from_dataframe(df_clean, value_cols=target_cols, freq="D", fill_missing_dates=True, fillna_value=0)
    past_ts = TimeSeries.from_dataframe(
        df_clean, value_cols=past_cov_cols, freq="D", fill_missing_dates=True, fillna_value=0
    )

    #Membuat Future Covariates sampai 60 hari ke depan
    df_covariates = load_holiday(start_date, extended_end_date)
    
    future_ts = TimeSeries.from_dataframe(
        df_covariates, value_cols=["hari_libur"], freq="D", fill_missing_dates=True, fillna_value=0
    )

    #Scaling
    scaler_y, scaler_past, scaler_future = Scaler(), Scaler(), Scaler()
    y_scaled = scaler_y.fit_transform(y_ts)
    past_scaled = scaler_past.fit_transform(past_ts).slice_intersect(y_scaled)
    future_scaled = scaler_future.fit_transform(future_ts)

    return (
        y_scaled,
        past_scaled,
        future_scaled,
        scaler_y,
        df_covariates,
    )

@st.cache_data
def load_recipe():
    #Memuat data resep & rasio bahan baku
    df = pd.read_csv(gsheet_resep)

    #Hitung rasio pemakaian bahan baku per total produksi
    kol_bahan = [col for col in df.columns if col != 'Produksi']
    df_rasio = df[kol_bahan].div(df['Produksi'], axis=0)
    rasio_bahan_mean = df_rasio.mean()
    return rasio_bahan_mean

def load_holiday(_sales_index):
    df = pd.read_csv(gholiday_url, index_col='event')

    # Normalisasi tanggal input
    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    ext_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    # Buat set tanggal berformat datetime.date
    workday_dates = set()
    for _, row in df[df["status"] == "workday"].iterrows():
        if pd.notna(row["start"]) and pd.notna(row["end"]):
            workday_dates.update(pd.date_range(row["start"], row["end"]).date)

    cust_hol_dates = set()
    for _, row in df.iterrows():
        if pd.notna(row["start"]) and pd.notna(row["end"]):
            cust_hol_dates.update(pd.date_range(row["start"], row["end"]).date)

    years_list = [int(y) for y in ext_dates.year.unique()]
    id_holidays = holidays.country_holidays('ID', years=years_list)

    df_covariates = pd.DataFrame(index=ext_dates)

    def cek_status_libur(ts):
        d = ts.date()  # KONVERSI KUNCI: Ubah Timestamp ke datetime.date
        if d in workday_dates:
            return 0
        if d in id_holidays:
            return 1
        if d in cust_hol_dates:
            return 1
        if ts.day_of_week == 6 and ts.isocalendar().week % 2 == 0:
            return 1
        return 0

    df_covariates["hari_libur"] = df_covariates.index.map(cek_status_libur)
    return df_covariates
    
@st.cache_resource
def load_model():
    #Memuat model TFT yang sudah dilatih
    return TFTModel.load(model_path)

def hitung_wmape_manual(y_true_ts, y_pred_ts):
    # Ambil array nilai asli dan prediksi
    actual = y_true_ts.values().flatten()
    pred = y_pred_ts.values().flatten()

    # Rumus WMAPE: (Sum Absolute Error / Sum Actual) * 100
    sum_actual = np.sum(actual)
    return(
        0.0 
        if sum_actual == 0
        else (np.sum(np.abs(actual - pred)) / sum_actual)* 100
    )

@st.cache_data
def load_metrics(_y_ts, _pred_terjual, target_cols):
    try:
        mae_list =[]
        rmse_list =[]
        wmape_list=[]
        
        for col in target_cols:
            y_true_single = _y_ts[col].slice_intersect(_pred_terjual)
            y_pred_single = _pred_terjual[col]

            mae_list.append(mae(y_true_single, y_pred_single))
            rmse_list.append(rmse(y_true_single, y_pred_single))
            wmape_list.append(hitung_wmape_manual(y_true_single, y_pred_single))

        metrics_df = pd.DataFrame({
            "metric": ["MAE", "RMSE", "WMAPE"],
            "value": [
                np.mean(mae_list),
                np.mean(rmse_list),
                np.mean(wmape_list)
            ],
        })

        return metrics_df

    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def compute_evaluation_metrics(_model, df_sales, target_cols):
    try:
        df_eval = df_sales.tail(60).copy()

        y_ts_eval = TimeSeries.from_dataframe(
            df_eval, value_cols=target_cols, fill_missing_dates=True, freq="D", fillna_value=0
        )

        past_cov_ts = TimeSeries.from_dataframe(
            df_eval, value_cols=["Sisa"], fill_missing_dates=True, freq="D", fillna_value=0
        )

        eval_ext_index = pd.date_range(
            start = df_eval.index.min(),
            end= df_eval.index.max() + pd.Timedelta(days=30),
            freq= "D"
        )
        df_cov = load_holiday(eval_ext_index)
        
        future_cov_ts = TimeSeries.from_dataframe(
            df_cov,
            value_cols=["hari_libur"],
            fill_missing_dates=True,
            freq="D",
        )
    
        scaler_y = Scaler()
        scaler_past = Scaler()
        scaler_future = Scaler()

        y_scaled = scaler_y.fit_transform(y_ts_eval)
        past_conv_scaled = scaler_past.fit_transform(past_cov_ts)
        future_conv_scaled = scaler_future.fit_transform(future_cov_ts)

        historical_pred_scaled = _model.historical_forecasts(
             series=y_scaled,
             past_covariates=past_conv_scaled,
             future_covariates=future_conv_scaled,
             start=0.5,
             forecast_horizon=1,
             stride=1,
             retrain=False,
             verbose=False,
        )

        historical_pred = scaler_y.inverse_transform(historical_pred_scaled)

        return load_metrics(y_ts_eval, historical_pred, target_cols)

    except Exception as e:
        print(f"Error pada evaluasi model: {e}")
        return None
        
#Helper telegram
def send_telegram_sync(message):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot = Bot(token=telegram_token)
        loop.run_until_complete(
            bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        )
        loop.close()
        return True

    except Exception as e:
        st.error(f"Gagal kirim Telegram: {e}")
        return False

#============================
#MAIN INTERFACE
#============================

st.title("🥖 SmartStock AI - Prediksi Stok Risol Mayo Mecca")
st.caption("AI Forecasting Penjualan dan Kebutuhan Bahan Baku Harian")

try:
    df_sales = load_sales()
    rasio_resep = load_recipe()
    model = load_model()
    st.success(f"Data tersinkronisasi: {len(df_sales)} hari transaksi historis")
except Exception as e:
    st.error(f"Gagal memuat resource: {e}")
    st.stop()

#================================
#SIDEBAR INPUT REAL-TIME HARI INI
#================================
st.sidebar.header("📥 Input Penjualan Hari Ini")

#SOP & Validasi tanggal
#Mencegah menginput total penjualan harian dua kali
#menggunakan zona waktu lokal (WIB), menghindari jam UTC server
tz = pytz.timezone("Asia/Jakarta")
tanggal_hari_ini = datetime.now(tz).date()
tanggal_terakhir_db = df_sales.index.max().date()

#Pengecekan apakah data hari ini sudah pernah disimpan
sudah_input = tanggal_terakhir_db >= tanggal_hari_ini

if sudah_input:
    st.sidebar.warning(
        f"Data penjualan hari ini (**{tanggal_hari_ini.strftime('%Y-%m-%d')}**) sudah diinput"
    )
    st.sidebar.caption(
        "**SOP:** Input baru hanya dapat dilakukan esok hari setelah penjualan selesai"
    )
else:
    tanggal_baru = tanggal_hari_ini
    st.sidebar.info(f"Tanggal Input: **{tanggal_baru.strftime('%Y-%m-%d')}**")
    st.sidebar.caption("**SOP:** Masukkan total penjualan setelah penjualan selesai")
    
#input total produksi
prod_ayam = st.sidebar.number_input("Produksi Ayam", min_value=0)
prod_udang = st.sidebar.number_input("Produksi Udang", min_value=0)
prod_keju = st.sidebar.number_input("Produksi Keju", min_value=0)
prod_telur = st.sidebar.number_input("Produksi Telur", min_value=0)
prod_sosis = st.sidebar.number_input("Produksi Sosis", min_value=0)
tot_prod = prod_ayam +  prod_udang + prod_keju + prod_telur + prod_sosis

#input penjualan 5 menu utama
input_ayam = st.sidebar.number_input("Terjual Ayam", min_value=0)
input_udang = st.sidebar.number_input("Terjual Udang", min_value=0)
input_keju = st.sidebar.number_input("Terjual Keju", min_value=0)
input_telur = st.sidebar.number_input("Terjual Telur", min_value=0)
input_sosis = st.sidebar.number_input("Terjual Sosis", min_value=0)
tot_sale = input_ayam + input_udang + input_keju + input_telur + input_sosis

#Input sisa stok
sisa = tot_prod - tot_sale
st.sidebar.metric("Sisa Stok (Semua Menu)", value=sisa)

button_predict = st.sidebar.button("🚀 Simpan & Prediksi Produksi Besok", disabled=sudah_input)

#Tampilan dashboards metrics
st.subheader("Monitoring Performa Model")

#persiapan timeseries dan scaling
target_cols = [
    "Terjual Ayam",
    "Terjual Udang",
    "Terjual Keju",
    "Terjual Telur",
    "Terjual Sosis"
]

with st.spinner("Menghitung evaluasi performa model..."):
    metrics = compute_evaluation_metrics(model, df_sales, target_cols)

if metrics is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics.loc[metrics.metric == 'MAE', 'value'].values[0]:.2f} pcs")
    c2.metric("RMSE", f"{metrics.loc[metrics.metric == 'RMSE', 'value'].values[0]:.2f}")
    c3.metric("WMAPE", f"{metrics.loc[metrics.metric == 'WMAPE', 'value'].values[0]:.2f}%")
else:
    st.info("Metric evaluasi belum tersedia")

#=============================
#Forecasting Logic
#============================
if button_predict and not sudah_input:
    start_time = time.time()

    #menyimpan ke google sheets
    with st.spinner("Menyimpan data hari ini ke Google Sheets"):
        row_to_save= [
            tanggal_baru.strftime("%Y-%m-%d"),
            tanggal_baru.strftime("%A"), #untuk mendapatkan nama hari
            prod_ayam,
            prod_udang,
            prod_keju,
            prod_telur,
            prod_sosis,
            tot_prod,
            input_ayam,
            input_udang,
            input_keju,
            input_telur,
            input_sosis,
            tot_sale,
            sisa
        ]

        try:
            #proses simpan data ke googlesheets
            append_data_to_gsheet(SPREADSHEET_ID, row_to_save)
            #DataFrmae lokal instan agar menghindari delay ekspor CSV dari Google Sheet
            #Update Dataframe sales dengan data input hari ini
            today_index = pd.to_datetime(tanggal_baru)
            today_data = pd.DataFrame([{
                "Tanggal": tanggal_baru.strftime("%Y-%m-%d"),
                "Hari": tanggal_baru.strftime("%A"),
                "Produksi Ayam": prod_ayam,
                "Produksi Udang": prod_udang,
                "Produksi Keju": prod_keju,
                "Produksi Telur": prod_telur,
                "Produksi Sosis": prod_sosis,
                "Total Produksi": tot_prod,
                "Terjual Ayam": input_ayam,
                "Terjual Udang": input_udang,
                "Terjual Keju": input_keju,
                "Terjual Telur": input_telur,
                "Terjual Sosis": input_sosis,
                "Total Terjual": tot_sale,
                "Sisa": sisa
            }], index=pd.DatetimeIndex([today_index]))
            
            df_sales_copy = df_sales.copy()
            df_sales_copy.index = pd.to_datetime(df_sales_copy.index)

            df_total = pd.concat([df_sales_copy, today_data])
            df_total.index = pd.to_datetime(df_total.index).normalize()
            df_total = df_total[~df_total.index.duplicated(keep='last')].sort_index()
            
            #Clear cache untuk pemanggilan berikutnya
            load_sales.clear()
            st.toast(
                "Data hari ini berhasil tersimpan di Google Sheet!", icon="💾"
            )
        except Exception as e:
            st.error(
                f"Gagal menyimpan data ke Google Sheets! Pastikan st.secrets sudah dikonfigurasi. Detail Error: {e}"
            )
            st.stop()

    # Jalankan prediksi untuk besok
    with st.spinner("Perhitungan estimasi demand dan bahan baku"):
        besok_date = tanggal_baru + pd.Timedelta(days=1)

        #memnaggil fungsi helper dasrts
        (y_scaled,
         past_conv_scaled,
         future_conv_scaled,
         scaler_y,
         df_covariates,
        ) = prepare_darts_data(df_total, target_cols, ["Sisa"])

        required_length = model.input_chunk_length

        if len(y_scaled) < required_length:
            st.error(
                f"⚠️ **Gagal Melakukan Prediksi:**\n\n"
                f"Model membutuhkan minimal **{required_length} hari** data historis berturut-turut, "
                f"tetapi data yang dikirim saat ini hanya berisi **{len(y_scaled)} hari**."
            )
        else:
            try:
                #Model Prediction (Prediksi 1 hari ke depan n=1)
                future_pred_scaled = model.predict(
                    n=1,
                    series=y_scaled,
                    past_covariates=(
                        past_conv_scaled if past_conv_scaled is not None else None),
                    future_covariates=(
                        future_conv_scaled if future_conv_scaled is not None else None),
                )
            
                #Inverse scalling & safety buffer
                df_pred = (scaler_y.inverse_transform(future_pred_scaled).to_dataframe().clip(lower=0))
                df_rekom_menu = (df_pred * SAFETY_BUFFER).round().astype(int)

                #Mengecek besok libur/tidak
                besok_is_libur = df_covariates.loc[besok_date, "hari_libur"] == 1
                if besok_is_libur:
                     df_rekom_menu.loc[:, :] =0
                     status_toko = "TUTUP"
                else:
                     status_toko = "BUKA"

                #hitung rekomendasi Stok bahan baku
                total_pcs_produksi = df_rekom_menu.sum(axis=1).values[0]
                stok_bahan = (rasio_resep * total_pcs_produksi).round(2)

                duration = round(time.time() - start_time, 2)

                #============================
                #Tampilan Output
                #============================
                st.markdown("---")
                st.subheader(
                    f"Rekomendasi Produksi & Bahan Baku ({besok_date.strftime('%d %B %Y')})"
                )
                st.caption(f"Status Toko: **{status_toko}** | AI Inference: {duration} detik")

                #Grid 1: Produksi Menu(Pcs)
                st.markdown("##### Rekomendasi Produksi Menu (Pcs)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Ayam", f"{df_rekom_menu['Terjual Ayam'].iloc[0]} pcs")
                m2.metric("Udang", f"{df_rekom_menu['Terjual Udang'].iloc[0]} pcs")
                m3.metric("Keju", f"{df_rekom_menu['Terjual Keju'].iloc[0]} pcs")
                m4.metric("Telur", f"{df_rekom_menu['Terjual Telur'].iloc[0]} pcs")
                m5.metric("Sosis", f"{df_rekom_menu['Terjual Sosis'].iloc[0]} pcs")

                st.divider()
             
                #Ekstraksi nilai bahan baku secara amana dari Pandas Series/Dictionary
                ayam_kg=(
                    stok_bahan["Ayam (kg)"]
                    if "Ayam (kg)" in stok_bahan
                    else stok_bahan.get("Ayam", 0)
                )
                udang_kg=(
                    stok_bahan["Udang (kg)"]
                    if "Udang (kg)" in stok_bahan
                    else stok_bahan.get("Udang", 0)
                )
                sosis_pcs=(
                    stok_bahan["Sosis (pcs)"]
                    if "Sosis (pcs)" in stok_bahan
                    else stok_bahan.get("Sosis", 0)
                )
                keju_kg=(
                    stok_bahan["Keju (kg)"]
                    if "Keju (kg)" in stok_bahan
                    else stok_bahan.get("Keju", 0)
                )
                telur_btr=(
                    stok_bahan["Telur (butir)"]
                    if "Telur (butir)" in stok_bahan
                    else stok_bahan.get("Telur", 0)
                )
                tepung_kg=(
                    stok_bahan["Tepung (kg)"]
                    if "Tepung (kg)" in stok_bahan
                    else stok_bahan.get("Tepung", 0)
                )
                mentega_kg=(
                    stok_bahan["Mentega (kg)"]
                    if "Mentega (kg)" in stok_bahan
                    else stok_bahan.get("Mentega", 0)
                )
                mayonaise_kg=(
                    stok_bahan["Mayonaise (kg)"]
                    if "Mayonaise (kg)" in stok_bahan
                    else stok_bahan.get("Mayonaise", 0)
                )
                panir_kg=(
                    stok_bahan["Tepung Panir (kg)"]
                    if "Tepung Panir (kg)" in stok_bahan
                    else stok_bahan.get("Panir", 0)
                )
                kentang_wortel_kg=(
                    stok_bahan["Kentang Wortel (kg)"]
                    if "Kentang Wortel (kg)" in stok_bahan
                    else stok_bahan.get("Kentang_Wortel", 0)
                )
                seledri_kg=(
                    stok_bahan["Seledri (kg)"]
                    if "Seledri (kg)" in stok_bahan
                    else stok_bahan.get("Seledri", 0)
                )
                daun_bawang_kg=(
                    stok_bahan["Daun Bawang (kg)"]
                    if "Daun Bawang (kg)" in stok_bahan
                    else stok_bahan.get("Daun_bawang", 0)
                )
                bamer_kg=(
                    stok_bahan["Bawang Merah (kg)"]
                    if "Bawang Merah (kg)" in stok_bahan
                    else stok_bahan.get("Bamer", 0)
                )
                baput_kg=(
                    stok_bahan["Bawang Putih (kg)"]
                    if "Bawang Putih (kg)" in stok_bahan
                    else stok_bahan.get("Baput", 0)
                )

                #Belanja Bahan Baku
                st.markdown("##### Estimasi Kebutuhan Bahan Baku Utama")
                b1, b2, b3, b4, b5, b6, b7 = st.columns(7) 
                b1.metric("Ayam", f"{ayam_kg:.2f} Kg")
                b2.metric("Udang", f"{udang_kg:.2f} Kg")
                b3.metric("Sosis", f"{sosis_pcs} pcs")
                b4.metric("Keju", f"{keju_kg:.2f} Kg")
                b5.metric("Telur", f"{telur_btr} butir")
                b6.metric("Tepung", f"{tepung_kg:.2f} Kg")
                b7.metric("Mentega", f"{mentega_kg:.2f} Kg")

                b8, b9, b10, b11, b12, b13, b14 = st.columns(7)
                b8.metric("Mayonaise", f"{mayonaise_kg:.2f} Kg")
                b9.metric("Panir", f"{panir_kg:.2f} Kg")
                b10.metric("Kentang_Wortel", f"{kentang_wortel_kg:.2f} Kg")
                b11.metric("Seledri", f"{seledri_kg:.2f} Kg")
                b12.metric("Daun_bawang", f"{daun_bawang_kg:.2f} Kg")
                b13.metric("Bamer", f"{bamer_kg:.2f} Kg")
                b14.metric("Baput", f"{baput_kg:.2f} Kg")

                #================
                #mengirim notifikasi telegram
                #================
                if telegram_token and chat_id:
                    pesan_telegram= textwrap.dedent(f"""
                    *SMARTSTOCK AI - NOTIFIKASI PRODUKSI*
                    Target Tanggal: *{besok_date.strftime('%d %B %Y')}*
                    Status Toko: *{status_toko}*

                    *Rekomendasi Produksi Menu:*
                    - Ayam: {df_rekom_menu['Terjual Ayam'].iloc[0]} pcs
                    - Udang: {df_rekom_menu['Terjual Udang'].iloc[0]} pcs
                    - Keju: {df_rekom_menu['Terjual Keju'].iloc[0]} pcs
                    - Telur: {df_rekom_menu['Terjual Telur'].iloc[0]} pcs
                    - Sosis: {df_rekom_menu['Terjual Sosis'].iloc[0]} pcs

                    *Estimasi Belanja Bahan:*
                    - Ayam, {ayam_kg:.2f} Kg
                    - Udang, {udang_kg:.2f} Kg
                    - Sosis, {sosis_pcs} pcs
                    - Keju, {keju_kg:.2f} Kg
                    - Telur, {telur_btr} butir
                    - Tepung, {tepung_kg:.2f} Kg
                    - Mentega, {mentega_kg:.2f} Kg
                    - Mayonaise, {mayonaise_kg:.2f} Kg
                    - Panir, {panir_kg:.2f} Kg
                    - Kentang_Wortel, {kentang_wortel_kg:.2f} Kg
                    - Seledri, {seledri_kg:.2f} Kg
                    - Daun_bawang, {daun_bawang_kg:.2f} Kg
                    - Bamer, {bamer_kg:.2f} Kg
                    - Baput, {baput_kg:.2f} Kg

                   ⚡_Inference time: {duration}s_
                    """)
                    if send_telegram_sync(pesan_telegram):
                        st.toast("Notifikasi berhasil dikirim ke telegram!", icon="✅")
            except Exception as e:
                st.error(
                    f"⚠️ **Terjadi Kesalahan pada Model Prediction:** {e}\n\n*Tips:"
                    " Pastikan rentang data tidak ada tanggal yang terlewat.*"
                )
#========
#History Chart
#========
st.divider()
st.subheader("📈 Trend Penjualan Historis (30 Hari Terakhir)")
st.line_chart(df_sales[target_cols].tail(30))
