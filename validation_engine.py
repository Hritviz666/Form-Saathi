# validation_engine.py
"""
Phase 2 — Step 3: Validation Engine
Validates user input for each form field.

Two layers:
    1. Rule-based  — regex, format, range, on-device sensitive checks
    2. LLM cross-field — GPT-4o-mini checks logical consistency across fields

Sensitive fields (Aadhaar, account number, income):
    Validated on-device only. Raw value NEVER sent to LLM.
    Only pass/fail result is forwarded.

Usage:
    from validation_engine import validate_field, validate_form
    result = validate_field("pan", "ABCDE1234F")
    results = validate_form(filled_fields_dict)
"""

from __future__ import annotations

import re
import os
import json
import requests
from datetime import datetime, date
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL       = "gpt-4o-mini"
LLM_MAX_TOKENS  = 512
LLM_TEMPERATURE = 0.0    # deterministic — validation is not creative

IFSC_API_URL    = "https://ifsc.razorpay.com/{ifsc}"
IFSC_TIMEOUT    = 5      # seconds

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Result of validating one field.

    Attributes
    ----------
    field_name  : field identifier (e.g. "pan", "pincode")
    value       : value that was validated (masked for sensitive fields)
    valid       : True if validation passed
    error       : error message if invalid, None if valid
    suggestion  : corrective hint for the user (plain language, no jargon)
    sensitive   : True if raw value was never sent to LLM
    source      : "rule" | "llm" | "api"
    """
    field_name  : str
    value       : str
    valid       : bool
    error       : Optional[str]      = None
    suggestion  : Optional[str]      = None
    sensitive   : bool               = False
    source      : str                = "rule"

    def __repr__(self) -> str:
        status = "✓" if self.valid else "✗"
        return (f"[{status}] {self.field_name}: {self.value!r} "
                f"({'sensitive, ' if self.sensitive else ''}source={self.source})"
                + (f"\n    error: {self.error}" if self.error else "")
                + (f"\n    hint : {self.suggestion}" if self.suggestion else ""))


# ── Regex patterns ────────────────────────────────────────────────────────────

# PAN: 5 uppercase letters + 4 digits + 1 uppercase letter
PAN_REGEX       = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# Aadhaar: 12 digits (spaces allowed between groups)
AADHAAR_REGEX   = re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$")

# Pincode: 6 digits, cannot start with 0
PINCODE_REGEX   = re.compile(r"^[1-9][0-9]{5}$")

# IFSC: 4 uppercase letters + 0 + 6 alphanumeric
IFSC_REGEX      = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

# Phone: 10 digits, starts with 6-9
PHONE_REGEX     = re.compile(r"^[6-9][0-9]{9}$")

# Date: DD/MM/YYYY or DD-MM-YYYY
DATE_REGEX      = re.compile(r"^(\d{2})[\/\-](\d{2})[\/\-](\d{4})$")

# Name: only letters, spaces, dots, hyphens — no digits or special chars
NAME_REGEX      = re.compile(r"^[A-Za-z\s\.\-]+$")

# Email
EMAIL_REGEX     = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

# Account number: 9-18 digits
ACCOUNT_REGEX   = re.compile(r"^\d{9,18}$")


# ── Individual field validators ───────────────────────────────────────────────

def validate_pan(value: str) -> ValidationResult:
    """
    PAN format: AAAAA9999A
    4th character encodes entity type — validated for common types.
    Sensitive: False (PAN format check is structural, not private data in context)
    """
    v = value.strip().upper()

    if not v:
        return ValidationResult("pan", v, False,
            error="PAN cannot be empty",
            suggestion="Enter your 10-character PAN number, e.g. ABCDE1234F")

    if len(v) != 10:
        return ValidationResult("pan", v, False,
            error=f"PAN must be exactly 10 characters, got {len(v)}",
            suggestion="PAN format is: 5 letters + 4 digits + 1 letter. Example: ABCDE1234F")

    if not PAN_REGEX.match(v):
        return ValidationResult("pan", v, False,
            error="PAN format is invalid",
            suggestion="First 5 must be capital letters, next 4 must be digits, last 1 must be a capital letter.")

    # 4th character entity check
    ENTITY_CHARS = {"P": "Individual", "C": "Company", "H": "HUF",
                    "F": "Firm", "A": "AOP", "T": "Trust",
                    "B": "BOI", "L": "Local Authority", "J": "Artificial Juridical Person",
                    "G": "Government"}
    entity_char = v[3]
    if entity_char not in ENTITY_CHARS:
        return ValidationResult("pan", v, False,
            error=f"4th character '{entity_char}' is not a valid entity type",
            suggestion="The 4th letter in PAN represents the type of taxpayer. "
                       "For individuals it is 'P'. Check your PAN card.")

    return ValidationResult("pan", v, True,
        suggestion=f"Valid PAN — entity type: {ENTITY_CHARS[entity_char]}")


def validate_aadhaar(value: str) -> ValidationResult:
    """
    Aadhaar: 12 digits. SENSITIVE — raw value never sent to LLM.
    Returns masked value in result.
    """
    v = value.strip().replace(" ", "")
    masked = v[:4] + " XXXX XXXX" if len(v) >= 4 else "XXXX"

    if not v:
        return ValidationResult("aadhaar", masked, False,
            error="Aadhaar number cannot be empty",
            suggestion="Enter your 12-digit Aadhaar number",
            sensitive=True)

    if not v.isdigit():
        return ValidationResult("aadhaar", masked, False,
            error="Aadhaar must contain only digits",
            suggestion="Remove any letters or special characters from your Aadhaar number",
            sensitive=True)

    if len(v) != 12:
        return ValidationResult("aadhaar", masked, False,
            error=f"Aadhaar must be exactly 12 digits, got {len(v)}",
            suggestion="Your Aadhaar number printed on the card is always 12 digits",
            sensitive=True)

    if v[0] in ("0", "1"):
        return ValidationResult("aadhaar", masked, False,
            error="Aadhaar cannot start with 0 or 1",
            suggestion="Check your Aadhaar card — the number should not start with 0 or 1",
            sensitive=True)

    return ValidationResult("aadhaar", masked, True,
        suggestion="Aadhaar format is valid",
        sensitive=True)


def validate_pincode(value: str) -> ValidationResult:
    v = value.strip()

    if not v:
        return ValidationResult("pincode", v, False,
            error="Pincode cannot be empty",
            suggestion="Enter your 6-digit area pincode")

    if not PINCODE_REGEX.match(v):
        return ValidationResult("pincode", v, False,
            error="Invalid pincode format",
            suggestion="Pincode must be 6 digits and cannot start with 0. "
                       "Check the pincode on any postal envelope or India Post website.")

    return ValidationResult("pincode", v, True)


def validate_ifsc(value: str) -> ValidationResult:
    """
    IFSC: regex check first, then live Razorpay API call.
    Returns bank name and branch on success.
    """
    v = value.strip().upper()

    if not v:
        return ValidationResult("ifsc", v, False,
            error="IFSC code cannot be empty",
            suggestion="IFSC code is printed on your cheque book and passbook",
            source="rule")

    if not IFSC_REGEX.match(v):
        return ValidationResult("ifsc", v, False,
            error="IFSC format is invalid",
            suggestion="IFSC is 11 characters: first 4 letters are bank code, "
                       "5th is always 0, last 6 are branch code. Example: SBIN0001234",
            source="rule")

    # Live API check
    try:
        resp = requests.get(
            IFSC_API_URL.format(ifsc=v),
            timeout=IFSC_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            bank   = data.get("BANK", "Unknown Bank")
            branch = data.get("BRANCH", "Unknown Branch")
            city   = data.get("CITY", "")
            return ValidationResult("ifsc", v, True,
                suggestion=f"{bank} — {branch}, {city}",
                source="api")
        elif resp.status_code == 404:
            return ValidationResult("ifsc", v, False,
                error="IFSC code not found in RBI database",
                suggestion="Double-check the IFSC code on your cheque leaf or passbook. "
                           "Bank branches that have merged may have a new IFSC.",
                source="api")
        else:
            # API returned unexpected status — fall back to regex-only pass
            return ValidationResult("ifsc", v, True,
                suggestion="IFSC format is valid (live check unavailable)",
                source="rule")

    except requests.exceptions.Timeout:
        return ValidationResult("ifsc", v, True,
            suggestion="IFSC format is valid (live check timed out — verify manually)",
            source="rule")

    except requests.exceptions.RequestException:
        return ValidationResult("ifsc", v, True,
            suggestion="IFSC format is valid (offline — live check skipped)",
            source="rule")


def validate_dob(value: str,
                 min_age: int = 0,
                 max_age: int = 120) -> ValidationResult:
    """
    Date of birth: DD/MM/YYYY or DD-MM-YYYY.
    Checks format, calendar validity, and age range.
    """
    v = value.strip()

    if not v:
        return ValidationResult("dob", v, False,
            error="Date of Birth cannot be empty",
            suggestion="Enter date in DD/MM/YYYY format, e.g. 15/08/1990")

    m = DATE_REGEX.match(v)
    if not m:
        return ValidationResult("dob", v, False,
            error="Date format is invalid",
            suggestion="Use DD/MM/YYYY format. Example: 15/08/1990")

    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))

    try:
        dob = date(year, month, day)
    except ValueError:
        return ValidationResult("dob", v, False,
            error=f"{day}/{month}/{year} is not a valid calendar date",
            suggestion="Check the day and month — for example, there is no 31st February")

    today = date.today()
    if dob > today:
        return ValidationResult("dob", v, False,
            error="Date of Birth cannot be in the future",
            suggestion="Please enter your actual date of birth")

    age = (today - dob).days // 365
    if age < min_age:
        return ValidationResult("dob", v, False,
            error=f"Applicant must be at least {min_age} years old",
            suggestion=f"Minimum age requirement is {min_age} years")

    if age > max_age:
        return ValidationResult("dob", v, False,
            error=f"Age of {age} years is not plausible",
            suggestion="Please re-check the year of birth")

    return ValidationResult("dob", v, True,
        suggestion=f"Valid date — age {age} years")


def validate_phone(value: str) -> ValidationResult:
    v = value.strip().replace(" ", "").replace("-", "")

    if not v:
        return ValidationResult("phone", v, False,
            error="Phone number cannot be empty",
            suggestion="Enter your 10-digit mobile number")

    if not PHONE_REGEX.match(v):
        return ValidationResult("phone", v, False,
            error="Invalid Indian mobile number",
            suggestion="Mobile number must be 10 digits and start with 6, 7, 8, or 9")

    return ValidationResult("phone", v, True)


def validate_name(value: str, field_name: str = "name") -> ValidationResult:
    v = value.strip()

    if not v:
        return ValidationResult(field_name, v, False,
            error="Name cannot be empty",
            suggestion="Enter your full name as it appears on your ID proof")

    if len(v) < 2:
        return ValidationResult(field_name, v, False,
            error="Name is too short",
            suggestion="Enter your complete name")

    if not NAME_REGEX.match(v):
        return ValidationResult(field_name, v, False,
            error="Name contains invalid characters",
            suggestion="Name should contain only letters, spaces, dots, or hyphens. "
                       "No numbers or special characters allowed.")

    return ValidationResult(field_name, v, True)


def validate_email(value: str) -> ValidationResult:
    v = value.strip()

    if not v:
        return ValidationResult("email", v, False,
            error="Email cannot be empty",
            suggestion="Enter a valid email address, e.g. name@example.com")

    if not EMAIL_REGEX.match(v):
        return ValidationResult("email", v, False,
            error="Invalid email format",
            suggestion="Email must contain @ and a domain, e.g. name@example.com")

    return ValidationResult("email", v, True)


def validate_account_number(value: str) -> ValidationResult:
    """Bank account number. SENSITIVE — raw value never sent to LLM."""
    v = value.strip().replace(" ", "")
    masked = v[:4] + "X" * (len(v) - 4) if len(v) > 4 else "XXXX"

    if not v:
        return ValidationResult("account_number", masked, False,
            error="Account number cannot be empty",
            sensitive=True)

    if not ACCOUNT_REGEX.match(v):
        return ValidationResult("account_number", masked, False,
            error="Account number must be 9–18 digits",
            suggestion="Enter your bank account number without spaces or dashes",
            sensitive=True)

    return ValidationResult("account_number", masked, True,
        sensitive=True)


# ── Field router ──────────────────────────────────────────────────────────────

# Maps field name keywords → validator function
FIELD_VALIDATORS = {
    "pan"            : validate_pan,
    "aadhaar"        : validate_aadhaar,
    "pincode"        : validate_pincode,
    "zip"            : validate_pincode,
    "ifsc"           : validate_ifsc,
    "dob"            : validate_dob,
    "date_of_birth"  : validate_dob,
    "phone"          : validate_phone,
    "mobile"         : validate_phone,
    "email"          : validate_email,
    "name"           : validate_name,
    "first_name"     : lambda v: validate_name(v, "first_name"),
    "last_name"      : lambda v: validate_name(v, "last_name"),
    "middle_name"    : lambda v: validate_name(v, "middle_name"),
    "account_number" : validate_account_number,
    "account_no"     : validate_account_number,
}


def validate_field(field_name: str, value: str) -> ValidationResult:
    """
    Validate a single field by name.
    Falls through to a generic non-empty check if field type is unknown.

    Usage:
        result = validate_field("pan", "ABCDE1234F")
        result = validate_field("dob", "15/08/1990")
        result = validate_field("ifsc", "SBIN0001234")
    """
    key = field_name.lower().strip().replace(" ", "_")

    # Fuzzy match — check if any known key is a substring of field_name
    validator = FIELD_VALIDATORS.get(key)
    if validator is None:
        for k, fn in FIELD_VALIDATORS.items():
            if k in key or key in k:
                validator = fn
                break

    if validator:
        return validator(value)

    # Generic fallback: just check non-empty
    if not value.strip():
        return ValidationResult(field_name, value, False,
            error=f"{field_name} cannot be empty",
            suggestion=f"Please fill in the {field_name} field")

    return ValidationResult(field_name, value, True, source="rule")


# ── Batch form validator ──────────────────────────────────────────────────────

def validate_form(fields: dict[str, str]) -> dict[str, ValidationResult]:
    """
    Validate all fields in a form dict.

    Parameters
    ----------
    fields : {"pan": "ABCDE1234F", "dob": "15/08/1990", ...}

    Returns
    -------
    dict mapping field_name → ValidationResult
    """
    results = {}
    for field_name, value in fields.items():
        results[field_name] = validate_field(field_name, value)
    return results


# ── LLM cross-field validation ────────────────────────────────────────────────

# Fields that are sensitive — raw values replaced with [PASS] or [FAIL]
SENSITIVE_FIELDS = {"aadhaar", "account_number", "account_no", "income"}


def _build_cross_field_prompt(
    fields          : dict[str, str],
    rule_results    : dict[str, ValidationResult],
    form_type       : str = "government form",
) -> str:
    """
    Build the user prompt for LLM cross-field check.
    Sensitive field values are replaced with their pass/fail status.
    """
    lines = [f"Form type: {form_type}\n",
             "Fields filled by the user (sensitive values are hidden):\n"]

    for field_name, value in fields.items():
        key = field_name.lower()
        if any(s in key for s in SENSITIVE_FIELDS):
            rule_r  = rule_results.get(field_name)
            status  = "[PASS]" if (rule_r and rule_r.valid) else "[FAIL]"
            display = status
        else:
            display = value

        rule_r   = rule_results.get(field_name)
        validity = "valid" if (rule_r and rule_r.valid) else "invalid"
        lines.append(f"  {field_name}: {display!r}  [{validity}]")

    lines.append("\nCheck for logical consistency issues between fields.")
    lines.append("Examples: DOB suggests minor but form is for adults only; "
                 "gender mismatch with title (Mr/Mrs/Kumari); "
                 "single parent selected but both parents filled; "
                 "future date in DOB; state does not match pincode prefix.")
    lines.append("\nRespond ONLY in this JSON format:")
    lines.append("""
{
  "issues": [
    {
      "fields": ["field1", "field2"],
      "issue": "short description",
      "suggestion": "plain language fix for the user"
    }
  ],
  "overall_valid": true
}
If no issues found, return {"issues": [], "overall_valid": true}""")

    return "\n".join(lines)


def validate_cross_field(
    fields      : dict[str, str],
    form_type   : str = "government form",
) -> dict:
    """
    Run LLM cross-field validation using GPT-4o-mini.
    Sensitive fields are masked before sending to API.

    Parameters
    ----------
    fields    : {"pan": "ABCDE1234F", "dob": "15/08/1990", ...}
    form_type : e.g. "PAN Form 49A", "SBI Account Opening Form"

    Returns
    -------
    dict with keys: issues (list), overall_valid (bool)
    """
    if not OPENAI_API_KEY:
        print("[cross_field]  OPENAI_API_KEY not set — skipping LLM validation")
        return {"issues": [], "overall_valid": True}

    # Run rule-based first to get pass/fail for sensitive fields
    rule_results = validate_form(fields)

    prompt = _build_cross_field_prompt(fields, rule_results, form_type)

    try:
        client   = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model       = LLM_MODEL,
            temperature = LLM_TEMPERATURE,
            max_tokens  = LLM_MAX_TOKENS,
            messages    = [
                {
                    "role"   : "system",
                    "content": (
                        "You are a form validation assistant for Indian government "
                        "and bank forms. You check logical consistency between fields. "
                        "You never see sensitive values like Aadhaar or account numbers. "
                        "Respond only in the JSON format specified."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")

        result = json.loads(raw)
        result["source"] = "llm"
        return result

    except json.JSONDecodeError as e:
        print(f"[cross_field]  LLM returned invalid JSON: {e}")
        return {"issues": [], "overall_valid": True, "source": "llm_error"}

    except Exception as e:
        print(f"[cross_field]  LLM call failed: {e}")
        return {"issues": [], "overall_valid": True, "source": "llm_error"}


# ── Full validation (rule + cross-field) ──────────────────────────────────────

def validate_all(
    fields    : dict[str, str],
    form_type : str = "government form",
    run_llm   : bool = True,
) -> dict:
    """
    Complete validation: rule-based per-field + LLM cross-field.

    Returns
    -------
    {
        "field_results"  : {field_name: ValidationResult, ...},
        "cross_field"    : {"issues": [...], "overall_valid": bool},
        "all_valid"      : bool   # True only if all rules pass AND no cross-field issues
    }
    """
    field_results = validate_form(fields)

    cross = {"issues": [], "overall_valid": True}
    if run_llm:
        cross = validate_cross_field(fields, form_type)

    all_rule_pass = all(r.valid for r in field_results.values())
    all_valid     = all_rule_pass and cross.get("overall_valid", True)

    return {
        "field_results" : field_results,
        "cross_field"   : cross,
        "all_valid"     : all_valid,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FormSaathi Validation Engine — Test Run")
    print("=" * 60)

    # ── Rule-based tests ──────────────────────────────────────────
    test_cases = [
        # (field_name,      value,              expected_valid)
        ("pan",             "ABCHE1234F",        True),
        ("pan",             "ABCD1234F",         False),   # too short
        ("pan",             "ABCDE12340",        False),   # last char digit
        ("aadhaar",         "9876 5432 1098",    True),
        ("aadhaar",         "123456789012",      False),   # starts with 1
        ("pincode",         "110001",            True),
        ("pincode",         "011001",            False),   # starts with 0
        ("pincode",         "12345",             False),   # 5 digits
        ("dob",             "15/08/1990",        True),
        ("dob",             "31/02/1990",        False),   # invalid date
        ("dob",             "15/08/2090",        False),   # future
        ("phone",           "9876543210",        True),
        ("phone",           "1234567890",        False),   # starts with 1
        ("ifsc",            "SBIN0001234",       True),    # regex only (no API in test)
        ("ifsc",            "SBI00001234",       False),   # wrong format
        ("email",           "user@example.com",  True),
        ("email",           "notanemail",        False),
        ("account_number",  "123456789012",      True),
        ("account_number",  "12345",             False),   # too short
    ]

    passed = 0
    failed = 0

    for field_name, value, expected in test_cases:
        result = validate_field(field_name, value)
        status = "PASS" if result.valid == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  [{status}]  {field_name:<16} {value!r:<25} → valid={result.valid}")
        if result.error:
            print(f"         error     : {result.error}")
        if result.suggestion:
            print(f"         suggestion: {result.suggestion}")

    print(f"\nRule-based tests: {passed} passed, {failed} failed")

    # ── IFSC live API test ────────────────────────────────────────
    print("\n── IFSC Live API Test ──────────────────────────────────")
    ifsc_result = validate_ifsc("SBIN0000001")
    print(ifsc_result)

    # ── Cross-field LLM test ──────────────────────────────────────
    if OPENAI_API_KEY:
        print("\n── Cross-field LLM Test ────────────────────────────────")
        sample_fields = {
            "title"     : "Mr.",
            "first_name": "Priya",         # female name with male title — mismatch
            "dob"       : "15/08/1990",
            "gender"    : "Female",
            "pincode"   : "110001",
        }
        cross = validate_cross_field(sample_fields, form_type="PAN Form 49A")
        print(json.dumps(cross, indent=2))
    else:
        print("\n[skip]  Set OPENAI_API_KEY env variable to test cross-field LLM validation")

    print("\nNext step: run agent.py")