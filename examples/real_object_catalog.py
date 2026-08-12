"""Pure object-catalog resolution for the SAM 3 real-image pipeline.

The real pipeline uses the FoundationPose/CAD ``Old_name`` as its applied
object key.  ``Object_name`` is used only when a catalog row has no legacy
name.  User input may come from ``Object_name``, ``Old_name``, or
``Class_name``; the returned identity always preserves the exact CSV spelling
needed by reference and capture-manifest joins.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import unicodedata


REQUIRED_COLUMNS = ("No", "Object_name", "Class_name", "Old_name", "Year")
LOOKUP_COLUMNS = ("Object_name", "Old_name", "Class_name")


class RealObjectCatalogError(ValueError):
    """Raised when a catalog is invalid or an object cannot be resolved."""


@dataclass(frozen=True, slots=True)
class RealObjectIdentity:
    """One catalog row normalized to the real pipeline's applied key."""

    applied_key: str
    object_id: str
    catalog_object_name: str
    old_name: str | None
    key_namespace: str
    catalog_year: int
    object_catalog_sha256: str
    matched_field: str

    def metadata(self) -> dict[str, str | int | None]:
        """Return JSON-safe identity/provenance fields for SAM artifacts."""

        return {
            "object_name": self.applied_key,
            "object_id": self.object_id,
            "old_name": self.old_name,
            "catalog_object_name": self.catalog_object_name,
            "key_namespace": self.key_namespace,
            "catalog_year": self.catalog_year,
            "object_catalog_sha256": self.object_catalog_sha256,
        }


@dataclass(frozen=True, slots=True)
class _CatalogRow:
    number: int
    object_name: str
    object_id: str
    old_name: str
    catalog_year: int
    source_line: int

    @property
    def applied_key(self) -> str:
        return self.old_name or self.object_name

    @property
    def key_namespace(self) -> str:
        return "Old_name" if self.old_name else "Object_name"


def resolve_real_object(query: str, objects_metadata: str | Path) -> RealObjectIdentity:
    """Resolve a real-pipeline object from any supported catalog namespace.

    Lookup comparison is whitespace-insensitive, Unicode-normalized, and
    case-insensitive.  Returned values retain the catalog's exact spelling.
    The catalog is validated globally so a query can never silently select an
    identity from an ambiguous alias namespace.
    """

    normalized_query = _normalize(query)
    source_path = Path(objects_metadata).expanduser().resolve()
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise RealObjectCatalogError(
            f"Could not read objects metadata {source_path}: {exc}"
        ) from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RealObjectCatalogError(
            f"Objects metadata must be UTF-8: {source_path} ({exc})"
        ) from exc

    rows = _load_rows(text, source_path)
    indexes = _build_indexes(rows)
    matches: list[tuple[str, _CatalogRow]] = []
    for column in LOOKUP_COLUMNS:
        row = indexes[column].get(normalized_query)
        if row is not None:
            matches.append((column, row))
    if not matches:
        raise RealObjectCatalogError(
            f"Object {query!r} was not found by Object_name, Old_name, or "
            f"Class_name in {source_path}."
        )

    identities = {row.object_id for _, row in matches}
    if len(identities) != 1:
        details = ", ".join(
            f"{field}={row.object_name}/{row.object_id}" for field, row in matches
        )
        raise RealObjectCatalogError(
            f"Object {query!r} is ambiguous across catalog fields: {details}."
        )
    matched_field, row = matches[0]
    return RealObjectIdentity(
        applied_key=row.applied_key,
        object_id=row.object_id,
        catalog_object_name=row.object_name,
        old_name=row.old_name or None,
        key_namespace=row.key_namespace,
        catalog_year=row.catalog_year,
        object_catalog_sha256=digest,
        matched_field=matched_field,
    )


def _load_rows(text: str, source_path: Path) -> list[_CatalogRow]:
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        fieldnames = tuple((name or "").strip() for name in reader.fieldnames or ())
        if not fieldnames:
            raise RealObjectCatalogError(f"Objects metadata is empty: {source_path}.")
        named_fields = [name for name in fieldnames if name]
        duplicates = sorted(
            {name for name in named_fields if named_fields.count(name) > 1}
        )
        if duplicates:
            raise RealObjectCatalogError(
                f"Objects metadata {source_path} has duplicate columns: "
                f"{duplicates}."
            )
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise RealObjectCatalogError(
                f"Objects metadata {source_path} is missing columns: {missing}."
            )
        # Use the validated, stripped names for subsequent DictReader rows.
        reader.fieldnames = list(fieldnames)

        rows: list[_CatalogRow] = []
        for source_line, raw_row in enumerate(reader, start=2):
            values = {
                column: (raw_row.get(column) or "").strip()
                for column in REQUIRED_COLUMNS
            }
            for column in ("No", "Object_name", "Class_name", "Year"):
                if not values[column]:
                    raise RealObjectCatalogError(
                        f"Objects metadata row {source_line} has an empty {column}."
                    )
            try:
                number = int(values["No"])
            except ValueError as exc:
                raise RealObjectCatalogError(
                    f"Objects metadata row {source_line} has non-integer "
                    f"No={values['No']!r}."
                ) from exc
            if number <= 0:
                raise RealObjectCatalogError(
                    f"Objects metadata row {source_line} has non-positive "
                    f"No={number}."
                )
            expected_object_id = f"obj_{number:03d}"
            if values["Class_name"] != expected_object_id:
                raise RealObjectCatalogError(
                    f"Objects metadata row {source_line} maps No={number} to "
                    f"Class_name={values['Class_name']!r}; expected "
                    f"{expected_object_id!r}."
                )
            try:
                catalog_year = int(values["Year"])
            except ValueError as exc:
                raise RealObjectCatalogError(
                    f"Objects metadata row {source_line} has non-integer "
                    f"Year={values['Year']!r}."
                ) from exc
            if catalog_year not in {2024, 2025, 2026}:
                raise RealObjectCatalogError(
                    f"Objects metadata row {source_line} has Year={catalog_year}; "
                    "expected 2024, 2025, or 2026."
                )
            rows.append(
                _CatalogRow(
                    number=number,
                    object_name=values["Object_name"],
                    object_id=values["Class_name"],
                    old_name=values["Old_name"],
                    catalog_year=catalog_year,
                    source_line=source_line,
                )
            )
    except csv.Error as exc:
        raise RealObjectCatalogError(
            f"Could not parse objects metadata {source_path}: {exc}"
        ) from exc
    if not rows:
        raise RealObjectCatalogError(f"Objects metadata has no rows: {source_path}.")
    return rows


def _build_indexes(
    rows: list[_CatalogRow],
) -> dict[str, dict[str, _CatalogRow]]:
    indexes: dict[str, dict[str, _CatalogRow]] = {
        column: {} for column in LOOKUP_COLUMNS
    }
    numbers: dict[int, _CatalogRow] = {}
    for row in rows:
        existing_number = numbers.get(row.number)
        if existing_number is not None:
            raise RealObjectCatalogError(
                f"Duplicate No={row.number} at rows "
                f"{existing_number.source_line} and {row.source_line}."
            )
        numbers[row.number] = row
        values = {
            "Object_name": row.object_name,
            "Old_name": row.old_name,
            "Class_name": row.object_id,
        }
        for column, value in values.items():
            if not value:
                continue
            normalized = _normalize(value)
            existing = indexes[column].get(normalized)
            if existing is not None:
                raise RealObjectCatalogError(
                    f"Duplicate normalized {column} {value!r} at rows "
                    f"{existing.source_line} and {row.source_line}."
                )
            indexes[column][normalized] = row

    # A token may occupy multiple namespaces only when every occurrence refers
    # to the same Class_name. This mirrors the capture-side startup validator.
    for left_index, left_name in enumerate(LOOKUP_COLUMNS):
        for right_name in LOOKUP_COLUMNS[left_index + 1 :]:
            left = indexes[left_name]
            right = indexes[right_name]
            for normalized in left.keys() & right.keys():
                if left[normalized].object_id != right[normalized].object_id:
                    raise RealObjectCatalogError(
                        f"Catalog alias {normalized!r} maps to different "
                        f"objects across {left_name} and {right_name}."
                    )
    return indexes


def _normalize(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealObjectCatalogError("Object query must be a non-empty string.")
    return unicodedata.normalize("NFKC", value).strip().casefold()
