
# تطبيق ويب Streamlit لمزامنة الملفات مع Google Drive/Sheets
import streamlit as st
import re
import pandas as pd
from datetime import datetime, date
import pytz
import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import gspread
from gspread.utils import rowcol_to_a1

# 📁 الإعدادات العامة

FOLDER_ID = '1bNtXXxT6D8jivdVT-ynL0xMfUyYfonaB'  # ID المجلد الجذري
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1TvO_nb8UfHHb2NihLWZ1eVcQChnzFiWqBZeD2gc_D14/edit'
WORKSHEET_NAME = 'data4'

EMPLOYEE_LIST = ["1085","1403","1093","1088","1084","1087","1092","1402","1400","1094","1603","1604","1607","1086","1608","1610","1611","1612","1154","1151","1602"]
MIN_SIZE = 400 * 1024  # 400 KB
MAX_SIZE = 4000 * 1024 # 4 MB

# 🔐 تهيئة خدمات Google باستخدام key.json
def initialize_services():
    # ---
    # لرفع التطبيق على Streamlit Cloud:
    # أضف متغير سرّي باسم GOOGLE_CREDENTIALS في إعدادات Secrets
    # وضع فيه كامل محتوى ملف service account (key.json) كنص JSON
    # مثال: في صفحة secrets أضف
    # GOOGLE_CREDENTIALS = '{...json...}'
    # ---
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    if not creds_json:
        st.error("لم يتم العثور على متغير البيئة GOOGLE_CREDENTIALS. يرجى إضافته في إعدادات Secrets.")
        st.stop()
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=['https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/spreadsheets']
    )
    sheet = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service, sheet

# ─────────────────────────
def get_audio_files(drive_service, root_id, date_start, date_end, employee):
    rows = []
    query = f"'{root_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    subfolders = drive_service.files().list(q=query, fields="files(id,name)").execute().get('files', [])
    for folder in subfolders:
        if not re.fullmatch(r"\d{8}", folder["name"]):
            continue
        folder_dt = datetime.strptime(folder["name"], "%Y%m%d").date()
        if not (date_start <= folder_dt <= date_end):
            continue
        query = f"'{folder['id']}' in parents and trashed=false and name contains '.wav'"
        files = drive_service.files().list(q=query, fields="files(id,name,size)").execute().get('files', [])
        for f in files:
            name = f["name"].lower()
            if any(tok in name for tok in ("(", ")", "copy")):
                continue
            size = int(f.get("size", 0))
            if not (MIN_SIZE <= size <= MAX_SIZE):
                continue
            base = re.sub(r"\.wav$", "", f["name"], flags=re.I)
            parts = base.split("-")
            if len(parts) < 5:
                continue
            # فلترة حسب الموظف المحدد فقط
            if employee and (parts[2] != employee and parts[3] != employee):
                continue
            rows.append({
                "اسم الملف": base,
                "رابط الملف": f"https://drive.google.com/file/d/{f['id']}/view?usp=sharing",
            })
    df = pd.DataFrame(rows)
    return df

# ─────────────────────────
def get_sheet_data(sheet, url, ws_name):
    sheet_id = re.search(r"/d/([a-zA-Z0-9-_]+)", url).group(1)
    ws = sheet.open_by_key(sheet_id).worksheet(ws_name)

    try:
        df_sheet = pd.DataFrame(ws.get_all_records())
    except:
        df_sheet = pd.DataFrame()

    # تأكد من الأعمدة الأساسية فقط (بدون "حجم الملف")
    headers = ws.row_values(1)
    def _ensure(col, idx):
        if col not in headers:
            ws.update_cell(1, idx, col)
            headers.append(col)
            if not df_sheet.empty:
                df_sheet[col] = ""

    _ensure("اسم الملف", 1)
    _ensure("رابط الملف", 2)
    _ensure("تاريخ المزامنة", 3)

    if df_sheet.empty:
        df_sheet = pd.DataFrame(columns=headers)

    return df_sheet, ws

# ─────────────────────────
def find_new_files(df_drive, df_sheet):
    new_df = df_drive if df_sheet.empty else df_drive[~df_drive["اسم الملف"].isin(df_sheet["اسم الملف"])]
    return new_df

# ─────────────────────────
def append_new_files(ws, df_sheet, new_df):
    if new_df.empty:
        return 0
    new_df = new_df.copy()
    new_df["تاريخ المزامنة"] = datetime.now(pytz.timezone("Asia/Riyadh")).strftime("%Y-%m-%d")
    for col in df_sheet.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[df_sheet.columns]
    ws.append_rows(values=new_df.values.tolist(), value_input_option="USER_ENTERED")
    return len(new_df)

class FileNameParser:
    @staticmethod
    def remove_parentheses(text):
        return re.sub(r'\s*\(\d+\)$', '', text)

    def parse(self, file_name):
        file_name = self.remove_parentheses(file_name)
        parts = file_name.split("-")

        if len(parts) < 5:
            print(f"⚠️ اسم غير صالح للتفكيك: {file_name}")
            return None

        try:
            raw_id = parts[0]
            date = f"{raw_id[:4]}-{raw_id[4:6]}-{raw_id[6:8]}"
            time = f"{raw_id[8:10]}:{raw_id[10:12]}"

            number1 = parts[2].strip()
            number2 = parts[3].strip()

            if len(number1) <= 7:
                extension, mobile = number1, number2
            else:
                extension, mobile = number2, number1

            return {
                "تاريخ المكالمة": date,
                "وقت المكالمة": time,
                "ترميز": raw_id,
                "رقم التحويلة": extension,
                "رقم الجوال": mobile,
                "نوع المكالمة": parts[4].strip(),
            }
        except Exception as e:
            print(f"⚠️ فشل التحليل: {file_name} ({e})")
            return None

class SheetManager:
    def __init__(self, sheet_service, sheet_url, worksheet_name):
        self.sheet_service = sheet_service
        self.sheet_url = sheet_url
        self.worksheet_name = worksheet_name
        self.df = pd.DataFrame()
        self.worksheet = None
        self.updated_indices = []
        self.orig_date_col = None

    def load_data(self):
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', self.sheet_url)
        sheet_id = match.group(1)
        sheet = self.sheet_service.open_by_key(sheet_id)
        self.worksheet = sheet.worksheet(self.worksheet_name)

        try:
            self.df = pd.DataFrame(self.worksheet.get_all_records())
        except:
            self.df = pd.DataFrame(columns=["اسم الملف"])

        if 'تاريخ المكالمة' in self.df.columns:
            self.orig_date_col = self.df['تاريخ المكالمة'].astype(str).fillna('')
        else:
            self.orig_date_col = pd.Series([''] * len(self.df))

    # لا طباعة في Streamlit

    def get_unarchived_rows(self):
        if 'تاريخ المكالمة' in self.df.columns:
            return self.df[self.df['تاريخ المكالمة'] == '']
        return self.df

    def update_metadata(self, parser):
        unarchived = self.get_unarchived_rows()
        self.updated_indices = []

        if unarchived.empty or 'اسم الملف' not in self.df.columns:
            print("ℹ️ لا توجد صفوف تحتاج تحديث.")
            return

        idxs = unarchived.index.tolist()
        names = self.df.loc[idxs, 'اسم الملف'].astype(str).tolist()

        updated = 0
        for idx, name in zip(idxs, names):
            if not name.strip():
                continue
            parsed = parser.parse(name)
            if parsed:
                for key, value in parsed.items():
                    self.df.at[idx, key] = value
                self.updated_indices.append(idx)
                updated += 1

    # لا طباعة في Streamlit

    def write_back(self, batch_size=400):
        if self.df.empty:
        # لا طباعة في Streamlit
            return

        if not self.updated_indices:
        # لا طباعة في Streamlit
            return

        cols = list(self.df.columns)
        ncols = len(cols)

        if 'تاريخ المكالمة' not in self.df.columns:
            # لا طباعة في Streamlit
            date_col_idx = None
            date_col_pos = None
        else:
            date_col_pos = self.df.columns.get_loc('تاريخ المكالمة')
            date_col_idx = date_col_pos + 1

        new_date_rows = []
        existing_date_rows = []
        for i in sorted(self.updated_indices):
            had_value = False
            if self.orig_date_col is not None and i < len(self.orig_date_col):
                had_value = str(self.orig_date_col.iloc[i]).strip() != ''
            (existing_date_rows if had_value else new_date_rows).append(i)

        def build_blocks(indices):
            blocks, start, prev = [], None, None
            for i in indices:
                if start is None:
                    start = prev = i
                elif i == prev + 1:
                    prev = i
                else:
                    blocks.append((start, prev))
                    start = prev = i
            if start is not None:
                blocks.append((start, prev))
            return blocks

        updated_rows = 0

        # A) صفوف تاريخها كان فارغًا
        for s, e in build_blocks(new_date_rows):
            r = s
            while r <= e:
                end = min(e, r + batch_size - 1)
                values = (
                    self.df.iloc[r:end+1][cols]
                    .fillna('')
                    .astype(object)
                    .values
                    .tolist()
                )
                self.worksheet.update(
                    range_name=f"A{r + 2}:{rowcol_to_a1(end + 2, ncols)}",
                    values=values,
                    value_input_option='USER_ENTERED'
                )
                updated_rows += (end - r + 1)
                r = end + 1

        # B) صفوف تاريخها كان معبّأ
        if date_col_idx is not None and existing_date_rows:
            left_cols = cols[:date_col_pos]
            right_cols = cols[date_col_pos+1:]

            for s, e in build_blocks(existing_date_rows):
                r = s
                while r <= e:
                    end = min(e, r + batch_size - 1)

                    if left_cols:
                        left_values = (
                            self.df.iloc[r:end+1][left_cols]
                            .fillna('')
                            .astype(object)
                            .values
                            .tolist()
                        )
                        self.worksheet.update(
                            range_name=f"A{r + 2}:{rowcol_to_a1(end + 2, len(left_cols))}",
                            values=left_values,
                            value_input_option='USER_ENTERED'
                        )

                    if right_cols:
                        right_start_col = date_col_idx + 1
                        right_values = (
                            self.df.iloc[r:end+1][right_cols]
                            .fillna('')
                            .astype(object)
                            .values
                            .tolist()
                        )
                        self.worksheet.update(
                            range_name=f"{rowcol_to_a1(r + 2, right_start_col)}:{rowcol_to_a1(end + 2, ncols)}",
                            values=right_values,
                            value_input_option='USER_ENTERED'
                        )

                    updated_rows += (end - r + 1)
                    r = end + 1

    # لا طباعة في Streamlit

# ─────────────────────────

# 🚀 Streamlit Web App
def main():
    st.set_page_config(page_title="Google Drive Sync", layout="centered")
    st.title("مزامنة ملفات Google Drive مع Google Sheets")

    # كلمة السر
    password = st.text_input("كلمة السر", type="password")
    if password != "1234":
        st.warning("يرجى إدخال كلمة السر الصحيحة.")
        st.stop()

    # اختيار الموظف
    employee = st.selectbox("اختر رقم الموظف", options=[""] + EMPLOYEE_LIST, index=0)

    # اختيار التاريخ
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("تاريخ البداية", value=date(2025, 8, 5), format="YYYY-MM-DD")
    with col2:
        date_to = st.date_input("تاريخ النهاية", value=date(2025, 10, 20), format="YYYY-MM-DD")

    if date_from > date_to:
        st.error("تاريخ البداية يجب أن يكون قبل تاريخ النهاية.")
        st.stop()

    if st.button("مزامنة", type="primary"):
        with st.spinner("جاري المزامنة..."):
            try:
                drive_service, sheet = initialize_services()
                date_start, date_end = sorted([date_from, date_to])
                df_drive = get_audio_files(drive_service, FOLDER_ID, date_start, date_end, employee if employee else None)
                df_sheet, ws = get_sheet_data(sheet, SHEET_URL, WORKSHEET_NAME)
                new_files = find_new_files(df_drive, df_sheet)
                added_count = append_new_files(ws, df_sheet, new_files)

                # تحليل البيانات وتحديثها
                sheet_mgr = SheetManager(sheet, SHEET_URL, WORKSHEET_NAME)
                sheet_mgr.load_data()
                parser = FileNameParser()
                sheet_mgr.update_metadata(parser)
                sheet_mgr.write_back(batch_size=400)

                st.success(f"تمت المزامنة بنجاح!\nعدد الملفات الجديدة المضافة: {added_count}")
                if not df_drive.empty:
                    st.dataframe(df_drive)
                else:
                    st.info("لا توجد ملفات مطابقة للشروط.")
            except Exception as e:
                st.error(f"حدث خطأ أثناء المزامنة: {e}")

if __name__ == "__main__":
    main()