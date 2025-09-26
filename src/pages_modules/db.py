import sqlite3
import pandas as pd

DB_NAME = "spend_platform.db"

# Column mapping for spend_data
SPEND_COLUMN_MAP = {
    "Business Unit": "business_unit",
    "Region": "region",
    "Supplier Details": "supplier_details",
    "Invoice #": "invoice_no",
    "Invoice Item #": "invoice_item_no",
    "Invoice Item Type": "invoice_item_type",
    "Invoice Date": "invoice_date",
    "Item Description": "item_description",
    "Qty": "qty",
    "Unit of Measure": "unit_of_measure",
    "Currency": "currency",
    "Unit Price": "unit_price",
    "Total Amount": "total_amount"
}

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Spend Data
    c.execute("""CREATE TABLE IF NOT EXISTS spend_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    business_unit TEXT,
                    region TEXT,
                    supplier_details TEXT,
                    invoice_no TEXT,
                    invoice_item_no TEXT,
                    invoice_item_type TEXT,
                    invoice_date TEXT,
                    item_description TEXT NOT NULL,
                    qty INTEGER,
                    unit_of_measure TEXT,
                    currency TEXT,
                    unit_price REAL,
                    total_amount REAL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""")

    # Category Data
    c.execute("""CREATE TABLE IF NOT EXISTS category_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    "Category 1" TEXT NOT NULL,
                    "Category 2" TEXT,
                    "Category 3" TEXT,
                    "Category 4" TEXT,
                    "Category 5" TEXT
                )""")

    # Categorized Data
    c.execute("""CREATE TABLE IF NOT EXISTS categorized_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spend_id INTEGER,
                    item_description TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    category_1 TEXT NOT NULL,
                    category_2 TEXT NOT NULL,
                    category_3 TEXT NOT NULL,
                    category_4 TEXT NOT NULL,
                    category_5 TEXT NOT NULL,
                    confidence REAL,
                    FOREIGN KEY(spend_id) REFERENCES spend_data(id)
                )""")

    # Users
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )""")

    # Error Table (Error)
    c.execute("""CREATE TABLE IF NOT EXISTS error_table (
                    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_transaction TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    error_status TEXT,
                    error_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_resolved INTEGER DEFAULT 0,
                    resolved_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""")

    # Rules Table (Rules)
    c.execute("""CREATE TABLE IF NOT EXISTS rule_table (
                    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_description TEXT,
                    rule_data TEXT,
                    rule_condition TEXT,
                    active_flag INTEGER DEFAULT 1,
                    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_deleted INTEGER DEFAULT 0
                )""")

    conn.commit()
    conn.close()

def insert_data(table, df):
    conn = sqlite3.connect(DB_NAME)

    # 🟢 If inserting into spend_data → rename & align columns
    if table == "spend_data":
        df = df.rename(columns=SPEND_COLUMN_MAP)

        # Keep only valid columns defined in schema
        valid_cols = [
            "business_unit", "region", "supplier_details", "invoice_no",
            "invoice_item_no", "invoice_item_type", "invoice_date",
            "item_description", "qty", "unit_of_measure", "currency",
            "unit_price", "total_amount"
        ]
        df = df[[col for col in df.columns if col in valid_cols]]

    # Write to DB
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()
