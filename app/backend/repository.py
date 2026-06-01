from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProcessedScanInput:
    filename: str
    content: bytes
    detections: list[dict[str, Any]]


class AppRepository:
    def __init__(self, *, backend_dir: Path | None = None) -> None:
        self.backend_dir = (backend_dir or Path(__file__).resolve().parent).resolve()
        self.data_dir = self.backend_dir / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.db_path = self.data_dir / "app.db"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS studies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    study_date TEXT NOT NULL,
                    selected_model TEXT NOT NULL,
                    volume_mm3 REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS study_scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    detections_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sort_order INTEGER NOT NULL,
                    FOREIGN KEY (study_id) REFERENCES studies(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_studies_patient_id ON studies(patient_id);
                CREATE INDEX IF NOT EXISTS idx_studies_patient_date ON studies(patient_id, study_date);
                CREATE INDEX IF NOT EXISTS idx_study_scans_study_id ON study_scans(study_id);
                """
            )

    def list_patients(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.id,
                    p.first_name,
                    p.last_name,
                    p.created_at,
                    COUNT(s.id) AS study_count,
                    MAX(s.study_date) AS latest_study_date
                FROM patients p
                LEFT JOIN studies s ON s.patient_id = p.id
                GROUP BY p.id
                ORDER BY p.last_name COLLATE NOCASE ASC, p.first_name COLLATE NOCASE ASC, p.id ASC
                """
            ).fetchall()
            return [self._patient_row_to_dict(row) for row in rows]

    def create_patient(self, first_name: str, last_name: str) -> dict[str, Any]:
        created_at = utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO patients(first_name, last_name, created_at) VALUES (?, ?, ?)",
                (first_name.strip(), last_name.strip(), created_at),
            )
            patient_id = int(cur.lastrowid)
        patient = self.get_patient(patient_id)
        if patient is None:
            raise RuntimeError("Failed to read created patient")
        return patient

    def get_patient(self, patient_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    p.id,
                    p.first_name,
                    p.last_name,
                    p.created_at,
                    COUNT(s.id) AS study_count,
                    MAX(s.study_date) AS latest_study_date
                FROM patients p
                LEFT JOIN studies s ON s.patient_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (patient_id,),
            ).fetchone()
            return self._patient_row_to_dict(row) if row else None

    def list_studies_for_patient(self, patient_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.id,
                    s.patient_id,
                    s.study_date,
                    s.selected_model,
                    s.volume_mm3,
                    s.created_at,
                    COUNT(sc.id) AS scan_count
                FROM studies s
                LEFT JOIN study_scans sc ON sc.study_id = s.id
                WHERE s.patient_id = ?
                GROUP BY s.id
                ORDER BY s.study_date DESC, s.id DESC
                """,
                (patient_id,),
            ).fetchall()
            return [self._study_summary_row_to_dict(row) for row in rows]

    def list_volume_trend(self, patient_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id AS study_id, study_date, volume_mm3
                FROM studies
                WHERE patient_id = ?
                ORDER BY study_date ASC, id ASC
                """,
                (patient_id,),
            ).fetchall()
            return [
                {
                    "study_id": int(row["study_id"]),
                    "study_date": row["study_date"],
                    "volume_mm3": float(row["volume_mm3"]),
                }
                for row in rows
            ]

    def get_study(self, study_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._get_study_with_conn(conn, study_id)

    def create_study(
        self,
        *,
        patient_id: int,
        study_date: str,
        selected_model: str,
        scans: list[ProcessedScanInput],
        volume_mm3: float,
    ) -> dict[str, Any]:
        created_at = utc_now_iso()

        with self._connect() as conn:
            patient_exists = conn.execute(
                "SELECT 1 FROM patients WHERE id = ?",
                (patient_id,),
            ).fetchone()
            if patient_exists is None:
                raise ValueError("Patient not found")

            cur = conn.execute(
                """
                INSERT INTO studies(patient_id, study_date, selected_model, volume_mm3, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (patient_id, study_date, selected_model, volume_mm3, created_at),
            )
            study_id = int(cur.lastrowid)
            study_dir = self.uploads_dir / f"patient_{patient_id}" / f"study_{study_id}"
            study_dir.mkdir(parents=True, exist_ok=True)

            try:
                for index, scan in enumerate(scans):
                    suffix = Path(scan.filename).suffix.lower() or ".png"
                    stored_filename = f"{index + 1:03d}_{uuid4().hex}{suffix}"
                    stored_path = study_dir / stored_filename
                    stored_path.write_bytes(scan.content)
                    relative_path = stored_path.relative_to(self.uploads_dir).as_posix()

                    conn.execute(
                        """
                        INSERT INTO study_scans(
                            study_id,
                            filename,
                            stored_filename,
                            image_path,
                            detections_json,
                            created_at,
                            sort_order
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            study_id,
                            scan.filename,
                            stored_filename,
                            relative_path,
                            json.dumps(scan.detections),
                            created_at,
                            index,
                        ),
                    )
            except Exception:
                shutil.rmtree(study_dir, ignore_errors=True)
                raise

            study = self._get_study_with_conn(conn, study_id)

        if study is None:
            raise RuntimeError("Failed to read created study")
        return study

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _patient_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "full_name": f"{row['first_name']} {row['last_name']}",
            "created_at": row["created_at"],
            "study_count": int(row["study_count"] or 0),
            "latest_study_date": row["latest_study_date"],
        }

    def _study_summary_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "patient_id": int(row["patient_id"]),
            "study_date": row["study_date"],
            "selected_model": row["selected_model"],
            "volume_mm3": float(row["volume_mm3"]),
            "created_at": row["created_at"],
            "scan_count": int(row["scan_count"] or 0),
        }

    def _get_study_with_conn(
        self,
        conn: sqlite3.Connection,
        study_id: int,
    ) -> dict[str, Any] | None:
        study_row = conn.execute(
            """
            SELECT
                s.id,
                s.patient_id,
                s.study_date,
                s.selected_model,
                s.volume_mm3,
                s.created_at,
                p.first_name,
                p.last_name,
                COUNT(sc.id) AS scan_count
            FROM studies s
            INNER JOIN patients p ON p.id = s.patient_id
            LEFT JOIN study_scans sc ON sc.study_id = s.id
            WHERE s.id = ?
            GROUP BY s.id
            """,
            (study_id,),
        ).fetchone()
        if study_row is None:
            return None

        scan_rows = conn.execute(
            """
            SELECT id, filename, stored_filename, image_path, detections_json, created_at, sort_order
            FROM study_scans
            WHERE study_id = ?
            ORDER BY sort_order ASC, id ASC
            """,
            (study_id,),
        ).fetchall()

        return {
            "id": int(study_row["id"]),
            "patient_id": int(study_row["patient_id"]),
            "patient_name": f"{study_row['first_name']} {study_row['last_name']}",
            "study_date": study_row["study_date"],
            "selected_model": study_row["selected_model"],
            "volume_mm3": float(study_row["volume_mm3"]),
            "created_at": study_row["created_at"],
            "scan_count": int(study_row["scan_count"] or 0),
            "scans": [
                {
                    "id": int(scan_row["id"]),
                    "filename": scan_row["filename"],
                    "image_url": f"/storage/{scan_row['image_path']}",
                    "detections": json.loads(scan_row["detections_json"]),
                    "created_at": scan_row["created_at"],
                    "sort_order": int(scan_row["sort_order"]),
                }
                for scan_row in scan_rows
            ],
        }