"""
AI Receipt and Expense Tracker
Streamlit UI - Upload receipts, extract data with Gemma 4 E2B vision, track spending.
"""

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from PIL import Image

from receipt_expense_tracker.extractor import ReceiptExtractor
from receipt_expense_tracker.preprocessor import preprocess
from receipt_expense_tracker import (
    save_expense,
    load_expenses,
    delete_expense,
    get_category_totals,
)
from receipt_expense_tracker.database import CATEGORIES

st.set_page_config(
    page_title="Receipt Expense Tracker",
    page_icon="🧾",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Model - load once and cache in session state
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vision model, this may take a moment on first run...")
def load_model():
    return ReceiptExtractor()


extractor = load_model()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_upload, tab_dashboard = st.tabs(["Upload Receipt", "Dashboard"])


# ===========================================================================
# TAB 1: Upload and Extract
# ===========================================================================

with tab_upload:
    st.header("Upload a Receipt")
    st.caption(
        "Supported: photos of crumpled, low-light, or thermal-printed receipts. "
        "JPG, PNG, WEBP accepted."
    )

    uploaded = st.file_uploader(
        "Choose a receipt image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_img, col_form = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("Receipt Preview")
            original = Image.open(uploaded)
            st.image(original, width=600)

        with col_form:
            st.subheader("Extracted Data")

            if st.button("Extract with AI", type="primary", use_container_width=True):
                with st.spinner("Analysing receipt..."):
                    processed = preprocess(original)
                    result = extractor.extract(processed)
                st.session_state["extracted"] = result
                st.success("Extraction complete. Review and edit below before saving.")

            if "extracted" in st.session_state:
                data = st.session_state["extracted"]

                vendor = st.text_input("Vendor", value=data.get("vendor") or "")
                receipt_date = st.text_input(
                    "Date (YYYY-MM-DD)", value=data.get("date") or ""
                )
                category = st.selectbox(
                    "Category",
                    CATEGORIES,
                    index=CATEGORIES.index(data.get("category", "Other"))
                    if data.get("category") in CATEGORIES
                    else len(CATEGORIES) - 1,
                )

                st.markdown("**Line Items**")
                items = data.get("line_items") or []
                items_text = st.text_area(
                    "Line items (JSON)",
                    value=json.dumps(items, indent=2),
                    height=140,
                    help="Edit the JSON array directly if needed.",
                    label_visibility="collapsed",
                )

                col_sub, col_tax, col_total = st.columns(3)
                subtotal = col_sub.number_input(
                    "Subtotal", value=float(data.get("subtotal") or 0), step=0.01, format="%.2f"
                )
                tax = col_tax.number_input(
                    "Tax", value=float(data.get("tax") or 0), step=0.01, format="%.2f"
                )
                total = col_total.number_input(
                    "Total", value=float(data.get("total") or 0), step=0.01, format="%.2f"
                )

                if st.button("Save to Ledger", type="primary", use_container_width=True):
                    try:
                        parsed_items = json.loads(items_text)
                    except json.JSONDecodeError:
                        parsed_items = []

                    record = {
                        "vendor": vendor,
                        "date": receipt_date,
                        "line_items": parsed_items,
                        "subtotal": subtotal,
                        "tax": tax,
                        "total": total,
                        "category": category,
                    }
                    save_expense(record)
                    del st.session_state["extracted"]
                    st.success(f"Saved: {vendor} - ${total:.2f}")
                    st.rerun()


# ===========================================================================
# TAB 2: Dashboard
# ===========================================================================

with tab_dashboard:
    st.header("Expense Dashboard")

    # Filters
    col_start, col_end, col_cat = st.columns([1, 1, 1])
    default_start = date.today() - timedelta(days=30)
    start_date = col_start.date_input("From", value=default_start)
    end_date = col_end.date_input("To", value=date.today())
    cat_filter = col_cat.selectbox("Category", ["All"] + CATEGORIES)

    df = load_expenses(
        start_date=str(start_date),
        end_date=str(end_date),
        category=cat_filter if cat_filter != "All" else None,
    )

    # Summary metrics
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Spent", f"${df['total'].sum():.2f}" if not df.empty else "$0.00")
    m2.metric("Receipts", len(df))
    m3.metric(
        "Average per Receipt",
        f"${df['total'].mean():.2f}" if not df.empty else "$0.00",
    )
    st.divider()

    if df.empty:
        st.info("No expenses found for the selected filters. Upload a receipt to get started.")
    else:
        # Spending by category chart
        totals = get_category_totals(
            start_date=str(start_date),
            end_date=str(end_date),
        )
        if not totals.empty and (cat_filter == "All"):
            st.subheader("Spending by Category")
            st.bar_chart(totals.set_index("category")["total"])

        # Expense table
        st.subheader("Expense Ledger")
        display_cols = ["id", "date", "vendor", "category", "subtotal", "tax", "total"]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "subtotal": st.column_config.NumberColumn("Subtotal", format="$%.2f"),
                "tax": st.column_config.NumberColumn("Tax", format="$%.2f"),
                "total": st.column_config.NumberColumn("Total", format="$%.2f"),
            },
        )

        # Delete
        with st.expander("Delete an expense"):
            del_id = st.number_input("Expense ID to delete", min_value=1, step=1)
            if st.button("Delete", type="secondary"):
                delete_expense(int(del_id))
                st.success(f"Deleted expense #{del_id}")
                st.rerun()
