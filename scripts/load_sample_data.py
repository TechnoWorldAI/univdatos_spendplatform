"""
Data loader script to populate the database with sample spend data
"""

import os
import sys
from datetime import datetime

# Ensure repo root is importable when running this script directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import sqlite3
from src.utils.debug import debug_logger


def load_sample_data():
    """Load sample data from Excel files into the database"""

    # Initialize database connection
    conn = sqlite3.connect('spend_platform.db')
    cursor = conn.cursor()

    try:
        # Check if sample data file exists
        sample_file = 'context/sample spend data- filled.xlsx'
        if not os.path.exists(sample_file):
            debug_logger.error(f"Sample data file not found: {sample_file}")
            return

        # Read sample spend data
        debug_logger.info("Loading sample spend data...")
        df = pd.read_excel(sample_file)
        debug_logger.info(f"Loaded {len(df)} rows from sample data file")

        # Column mapping for database
        column_mapping = {
            'BU_CODE': 'bu_code',
            'BU _NAME': 'bu_name',
            'Region': 'region',
            'PO_NO': 'po_no',
            'PO_ITEM_NO': 'po_item_no',
            'Invoice Date': 'invoice_date',
            'SUPPLIER_NO': 'supplier_no',
            'SUPPLIER_NAME': 'supplier_name',
            'Invoice Header Amt - all items': 'invoice_amount',
            'CURRENCY_TYPE': 'currency_type',
            'UNIT_OF_PURCH': 'unit_of_purch',
            'Item Invoice Value': 'item_invoice_value',
            'Material Code': 'material_code',
            'Material Item Name': 'material_item_name',
            'Order description': 'order_description',
            'INV_ITEM_DESC': 'inv_item_desc',
            'PO Price': 'po_price',
            'PO purchase qty': 'po_purchase_qty',
            'Unit Price': 'unit_price',
            'Tower (Practice)': 'tower_practice',
            'Category': 'category',
            'Subcategory': 'subcategory'
        }

        # Rename columns
        df_mapped = df.rename(columns=column_mapping)

        # Select only available columns
        available_columns = [col for col in column_mapping.values() if col in df_mapped.columns]
        df_final = df_mapped[available_columns].copy()

        # Clean and convert data types
        if 'invoice_date' in df_final.columns:
            df_final['invoice_date'] = pd.to_datetime(df_final['invoice_date'], errors='coerce')

        # Handle numeric columns
        numeric_columns = ['invoice_amount', 'item_invoice_value', 'po_price', 'po_purchase_qty', 'unit_price']
        for col in numeric_columns:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

        # Remove existing data
        cursor.execute("DELETE FROM spend_transactions")

        # Insert new data
        df_final.to_sql('spend_transactions', conn, if_exists='append', index=False)
        debug_logger.info(f"Successfully inserted {len(df_final)} transactions into database")

        # Load unique vendors into vendors table
        debug_logger.info("Loading vendors...")
        if 'supplier_name' in df_final.columns:
            unique_vendors = df_final[['supplier_no', 'supplier_name']].drop_duplicates()
            unique_vendors = unique_vendors.dropna(subset=['supplier_name'])

            # Clear existing vendors
            cursor.execute("DELETE FROM vendors")

            # Insert vendors
            for _, row in unique_vendors.iterrows():
                supplier_no = row.get('supplier_no', '')
                supplier_name = row['supplier_name']
                normalized_name = supplier_name.strip().upper() if pd.notna(supplier_name) else ''

                cursor.execute('''
                    INSERT INTO vendors (vendor_name, normalized_name, supplier_no)
                    VALUES (?, ?, ?)
                ''', (supplier_name, normalized_name, supplier_no))

            debug_logger.info(f"Successfully inserted {len(unique_vendors)} vendors into database")

        # Load categories
        debug_logger.info("Loading categories...")
        categories_to_add = set()

        if 'category' in df_final.columns:
            categories_to_add.update(df_final['category'].dropna().unique())
        if 'subcategory' in df_final.columns:
            categories_to_add.update(df_final['subcategory'].dropna().unique())
        if 'tower_practice' in df_final.columns:
            categories_to_add.update(df_final['tower_practice'].dropna().unique())

        # Clear existing categories
        cursor.execute("DELETE FROM categories")

        # Insert categories
        for category in categories_to_add:
            if category and str(category).strip():
                cursor.execute('''
                    INSERT INTO categories (category_name, description)
                    VALUES (?, ?)
                ''', (str(category).strip(), f"Auto-generated from sample data"))

        debug_logger.info(f"Successfully inserted {len(categories_to_add)} categories into database")

        # Generate some sample errors for demonstration
        debug_logger.info("Generating sample error logs...")

        # Clear existing errors
        cursor.execute("DELETE FROM error_logs")

        # Create some sample errors
        sample_errors = [
            ("Negative Amount", "Transaction has negative invoice value", "Open"),
            ("Missing Supplier", "Supplier name is missing", "Open"),
            ("Invalid Date", "Invoice date is in invalid format", "Resolved"),
            ("Duplicate Transaction", "Potential duplicate transaction detected", "Open"),
            ("Missing Category", "Category information is missing", "Open")
        ]

        # Get some random transaction IDs
        cursor.execute("SELECT transaction_id FROM spend_transactions LIMIT 10")
        transaction_ids = [row[0] for row in cursor.fetchall()]

        for i, (error_type, description, status) in enumerate(sample_errors):
            if i < len(transaction_ids):
                cursor.execute('''
                    INSERT INTO error_logs (transaction_id, error_type, description, status)
                    VALUES (?, ?, ?, ?)
                ''', (transaction_ids[i], error_type, description, status))

        debug_logger.info(f"Successfully generated {len(sample_errors)} sample error logs")

        # Commit all changes
        conn.commit()
        debug_logger.info("✅ Sample data loading completed successfully!")

        # Print summary with optimized single query
        summary_query = """
            SELECT 
                (SELECT COUNT(*) FROM spend_transactions) as transactions_count,
                (SELECT COUNT(*) FROM vendors) as vendors_count,
                (SELECT COUNT(*) FROM categories) as categories_count,
                (SELECT COUNT(*) FROM error_logs) as errors_count
        """
        
        cursor.execute(summary_query)
        counts = cursor.fetchone()
        transactions_count, vendors_count, categories_count, errors_count = counts

        debug_logger.info("\n📊 Database Summary:")
        debug_logger.info(f"- Transactions: {transactions_count:,}")
        debug_logger.info(f"- Vendors: {vendors_count:,}")
        debug_logger.info(f"- Categories: {categories_count:,}")
        debug_logger.info(f"- Error Logs: {errors_count:,}")

    except Exception as e:
        debug_logger.exception(f"❌ Error loading sample data: {str(e)}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    load_sample_data()
