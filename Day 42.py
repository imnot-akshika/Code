import re
import csv
import json
from pathlib import Path

class DataCleaner:
    EMAIL_PATTERN = r"^[\w.-]+@[\w.-]+\.\w{2,}$"
    PHONE_PATTERN = r"^\+?[\d\s\-\(\)]{7,15}$"
    DATE_PATTERN  = r"^\d{4}-\d{2}-\d{2}$"

    def clean_name(self, name: str) -> str:
        return " ".join(name.strip().title().split())
    
    def validate_email(self, email: str) -> bool:
        EMAIL_PATTERN = r"^[\w.-]+@[\w.-]+\.\w{2,}$"
        return re.match(EMAIL_PATTERN, email)
    
    def validate_phone(self, phone: str) -> bool:
        PHONE_PATTERN = r"^\+?[\d\s\-\(\)]{7,15}$"
        return re.match(PHONE_PATTERN, phone)
    
    def extract_emails(self, text: str) -> list[str]:
        return re.findall(r"[\w.-]+@[\w.-]+\.\w{2,}", text)
    
    def extract_dates(self, text: str) -> list[str]:
        return re.findall(r"\d{4}-\d{2}-\d{2}", text)
    
    def clean_record(self, record: dict) -> tuple[dict, list[str]]:
        errors = []
        cleaned = {}

        cleaned["name"]  =self.clean_name(record.get("name", ""))

        email = record.get("email", "").lower().strip()
        cleaned["email"] = email
        if not self.validate_email(email):
            errors.append(f"Invalid email: {email}")

        phone = record.get("phone", "").strip()
        cleaned["phone"] = phone
        if not self.validate_phone(phone):
            errors.append(f"Invalid phone: {phone}")

        return cleaned, errors
    
    def process_csv(self, input_file: str, output_file: str) -> dict:
        processed = 0
        valid = 0
        invalid = 0
        all_errors = []

        with open(input_file) as infile:
            reader = csv.DictReader(infile)
            valid_rows = []

            for row in reader:
                processed += 1
                cleaned, errors = self.clean_record(row)
                if errors:
                    invalid += 1
                    all_errors.append(errors)
                else:
                    valid += 1
                    valid_rows.append(cleaned)

        with open(output_file, "w", newline="") as outfile:
            if valid_rows:
                writer = csv.DictWriter(outfile, fieldnames=valid_rows[0].keys())
                writer.writeheader()
                writer.writerows(valid_rows)

        return {
            "processed": processed,
            "valid": valid,
            "invalid": invalid,
            "errors": all_errors
        }


#exaample usage
cleaner = DataCleaner()

print(cleaner.clean_name("  alice   smith  "))
print(cleaner.validate_email("alice@example.com"))
print(cleaner.validate_email("not-an-email"))
print(cleaner.validate_phone("+44 7911 123456"))
print(cleaner.extract_emails("contact alice@example.com or bob@test.org"))
print(cleaner.extract_dates("Meeting on 2026-01-15 and follow up 2026-02-01"))

record = {"name": "  alice smith  ", "email": "ALICE@EXAMPLE.COM", "phone": "invalid"}
cleaned, errors = cleaner.clean_record(record)
print(cleaned)
print(errors)
