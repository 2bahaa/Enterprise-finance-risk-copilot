from decimal import Decimal

import pytest

from app.services.csv_processor import process_csv

HEADER = b"transaction_id,transaction_date,department,category,amount,currency,description\n"
VALID = HEADER + b"TXN-1,2026-01-05,IT,Hardware,100.50,usd,Laptop\n"


def test_valid_csv_is_parsed_and_normalised():
    file_hash, rows = process_csv(VALID)

    assert len(file_hash) == 64
    assert rows[0]["amount"] == Decimal("100.50")
    assert rows[0]["currency"] == "USD"


def test_same_content_gives_the_same_hash():
    assert process_csv(VALID)[0] == process_csv(VALID)[0]


def test_empty_file_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        process_csv(b"")


def test_missing_required_column_is_rejected():
    bad = (
        b"transaction_id,transaction_date,department,category,amount\n"
        b"TXN-1,2026-01-05,IT,Hardware,10.00\n"
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        process_csv(bad)


def test_amount_with_three_decimals_is_rejected():
    with pytest.raises(ValueError, match="Invalid amount"):
        process_csv(VALID.replace(b"100.50", b"100.505"))


def test_invalid_date_is_rejected():
    with pytest.raises(ValueError, match="Invalid transaction_date"):
        process_csv(VALID.replace(b"2026-01-05", b"05/01/2026"))


def test_duplicate_transaction_id_is_rejected():
    bad = VALID + b"TXN-1,2026-01-06,IT,Hardware,5.00,USD,Duplicate\n"
    with pytest.raises(ValueError, match="Duplicate transaction_id"):
        process_csv(bad)