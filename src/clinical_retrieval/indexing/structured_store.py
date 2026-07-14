from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from clinical_retrieval.schemas import Chunk, EncounterMeta
from clinical_retrieval.retrieval.query_parser import ParsedQuery


SCHEMA = """
CREATE TABLE IF NOT EXISTS encounters (
  encounter_id TEXT PRIMARY KEY,
  encounter_date TEXT,
  encounter_type TEXT,
  provider TEXT,
  facility TEXT,
  start_page INTEGER,
  end_page INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  page_start INTEGER,
  page_end INTEGER,
  section TEXT,
  encounter_id TEXT,
  encounter_date TEXT,
  chunk_type TEXT,
  table_type TEXT,
  provider TEXT
);
CREATE TABLE IF NOT EXISTS medications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT,
  encounter_id TEXT,
  encounter_date TEXT,
  medication TEXT,
  page_start INTEGER
);
CREATE TABLE IF NOT EXISTS laboratory_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT,
  encounter_id TEXT,
  encounter_date TEXT,
  lab TEXT,
  page_start INTEGER
);
CREATE TABLE IF NOT EXISTS diagnosis_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT,
  encounter_id TEXT,
  encounter_date TEXT,
  code TEXT,
  page_start INTEGER
);
CREATE TABLE IF NOT EXISTS vitals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT,
  encounter_id TEXT,
  encounter_date TEXT,
  vital TEXT,
  page_start INTEGER
);
CREATE TABLE IF NOT EXISTS referrals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT,
  encounter_id TEXT,
  encounter_date TEXT,
  specialty TEXT,
  page_start INTEGER
);
CREATE TABLE IF NOT EXISTS imaging_orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT,
  encounter_id TEXT,
  encounter_date TEXT,
  imaging TEXT,
  page_start INTEGER
);
CREATE TABLE IF NOT EXISTS providers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT,
  encounter_id TEXT,
  provider TEXT,
  page_start INTEGER
);
CREATE TABLE IF NOT EXISTS signatures (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT,
  encounter_id TEXT,
  encounter_date TEXT,
  provider TEXT,
  page_start INTEGER
);
CREATE TABLE IF NOT EXISTS exact_postings (
  term TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  PRIMARY KEY (term, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_date ON chunks(encounter_date);
CREATE INDEX IF NOT EXISTS idx_meds_name ON medications(medication);
CREATE INDEX IF NOT EXISTS idx_labs_name ON laboratory_results(lab);
CREATE INDEX IF NOT EXISTS idx_icd ON diagnosis_codes(code);
CREATE INDEX IF NOT EXISTS idx_postings_term ON exact_postings(term);
"""


class StructuredStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def rebuild(
        self,
        chunks: list[Chunk],
        encounters: list[EncounterMeta] | None = None,
    ) -> dict[str, int]:
        cur = self.conn.cursor()
        for table in (
            "encounters",
            "chunks",
            "medications",
            "laboratory_results",
            "diagnosis_codes",
            "vitals",
            "referrals",
            "imaging_orders",
            "providers",
            "signatures",
            "exact_postings",
        ):
            cur.execute(f"DELETE FROM {table}")

        if encounters:
            cur.executemany(
                """
                INSERT INTO encounters
                (encounter_id, encounter_date, encounter_type, provider, facility, start_page, end_page)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.encounter_id,
                        e.encounter_date,
                        e.encounter_type,
                        e.provider,
                        e.facility,
                        e.start_page,
                        e.end_page,
                    )
                    for e in encounters
                ],
            )

        chunk_rows = []
        med_rows = []
        lab_rows = []
        icd_rows = []
        vital_rows = []
        ref_rows = []
        img_rows = []
        prov_rows = []
        sig_rows = []

        for c in chunks:
            m = c.metadata
            chunk_rows.append(
                (
                    c.chunk_id,
                    m.page_start,
                    m.page_end,
                    m.section,
                    m.encounter_id,
                    m.encounter_date,
                    m.chunk_type,
                    m.table_type,
                    m.provider,
                )
            )
            ents = m.entities or {}
            for med in ents.get("medications", []) or []:
                med_rows.append(
                    (c.chunk_id, m.encounter_id, m.encounter_date, med, m.page_start)
                )
            for lab in ents.get("labs", []) or []:
                lab_rows.append(
                    (c.chunk_id, m.encounter_id, m.encounter_date, lab, m.page_start)
                )
            for code in ents.get("icd_codes", []) or []:
                icd_rows.append(
                    (c.chunk_id, m.encounter_id, m.encounter_date, code, m.page_start)
                )
            for bp in ents.get("blood_pressures", []) or []:
                vital_rows.append(
                    (c.chunk_id, m.encounter_id, m.encounter_date, f"BP {bp}", m.page_start)
                )
            for spec in ents.get("specialties", []) or []:
                ref_rows.append(
                    (c.chunk_id, m.encounter_id, m.encounter_date, spec, m.page_start)
                )
            for img in ents.get("imaging", []) or []:
                img_rows.append(
                    (c.chunk_id, m.encounter_id, m.encounter_date, img, m.page_start)
                )
            for prov in ents.get("providers", []) or []:
                prov_rows.append((c.chunk_id, m.encounter_id, prov, m.page_start))
            if m.table_type == "signature" or (
                m.section and "sign" in (m.section or "").lower()
            ):
                sig_rows.append(
                    (
                        c.chunk_id,
                        m.encounter_id,
                        m.encounter_date,
                        m.provider,
                        m.page_start,
                    )
                )

        cur.executemany(
            """
            INSERT INTO chunks
            (chunk_id, page_start, page_end, section, encounter_id, encounter_date,
             chunk_type, table_type, provider)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )
        cur.executemany(
            "INSERT INTO medications (chunk_id, encounter_id, encounter_date, medication, page_start) VALUES (?,?,?,?,?)",
            med_rows,
        )
        cur.executemany(
            "INSERT INTO laboratory_results (chunk_id, encounter_id, encounter_date, lab, page_start) VALUES (?,?,?,?,?)",
            lab_rows,
        )
        cur.executemany(
            "INSERT INTO diagnosis_codes (chunk_id, encounter_id, encounter_date, code, page_start) VALUES (?,?,?,?,?)",
            icd_rows,
        )
        cur.executemany(
            "INSERT INTO vitals (chunk_id, encounter_id, encounter_date, vital, page_start) VALUES (?,?,?,?,?)",
            vital_rows,
        )
        cur.executemany(
            "INSERT INTO referrals (chunk_id, encounter_id, encounter_date, specialty, page_start) VALUES (?,?,?,?,?)",
            ref_rows,
        )
        cur.executemany(
            "INSERT INTO imaging_orders (chunk_id, encounter_id, encounter_date, imaging, page_start) VALUES (?,?,?,?,?)",
            img_rows,
        )
        cur.executemany(
            "INSERT INTO providers (chunk_id, encounter_id, provider, page_start) VALUES (?,?,?,?)",
            prov_rows,
        )
        cur.executemany(
            "INSERT INTO signatures (chunk_id, encounter_id, encounter_date, provider, page_start) VALUES (?,?,?,?,?)",
            sig_rows,
        )

        # Exact-search posting lists (entity tokens → chunk ids)
        posting_rows: list[tuple[str, str]] = []
        for c in chunks:
            ents = c.metadata.entities or {}
            terms: set[str] = set()
            for key in ("medications", "icd_codes", "providers", "imaging", "specialties"):
                for v in ents.get(key, []) or []:
                    terms.add(str(v).lower())
            for lab in ents.get("labs", []) or []:
                terms.add(str(lab).lower())
                # also index lab name token before value
                terms.add(str(lab).split()[0].lower())
            for bp in ents.get("blood_pressures", []) or []:
                terms.add(str(bp).lower())
            for t in terms:
                if len(t) >= 2:
                    posting_rows.append((t, c.chunk_id))
        cur.executemany(
            "INSERT OR IGNORE INTO exact_postings (term, chunk_id) VALUES (?, ?)",
            posting_rows,
        )
        self.conn.commit()
        return {
            "chunks": len(chunk_rows),
            "medications": len(med_rows),
            "labs": len(lab_rows),
            "icd": len(icd_rows),
            "signatures": len(sig_rows),
            "encounters": len(encounters or []),
            "exact_postings": len(posting_rows),
        }

    def load_posting_index(self) -> dict[str, list[str]]:
        """term → chunk_ids for exact_search candidate pruning."""
        cur = self.conn.cursor()
        try:
            rows = cur.execute("SELECT term, chunk_id FROM exact_postings").fetchall()
        except sqlite3.OperationalError:
            return {}
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["term"], []).append(r["chunk_id"])
        return out

    def structured_search(
        self, pq: ParsedQuery, top_k: int = 40
    ) -> list[tuple[str, float]]:
        """Filter-oriented clinical metadata retrieval."""
        scores: dict[str, float] = {}

        def bump(chunk_id: str, amount: float) -> None:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + amount

        cur = self.conn.cursor()

        # Date / year-month filters
        for d in pq.dates:
            if len(d) == 10:
                rows = cur.execute(
                    "SELECT chunk_id FROM chunks WHERE encounter_date = ?", (d,)
                ).fetchall()
                for r in rows:
                    bump(r["chunk_id"], 3.0)
                # signatures on that date are especially important
                sigs = cur.execute(
                    "SELECT chunk_id FROM signatures WHERE encounter_date = ?", (d,)
                ).fetchall()
                for r in sigs:
                    bump(r["chunk_id"], 4.0)
            elif len(d) == 7:
                rows = cur.execute(
                    "SELECT chunk_id FROM chunks WHERE encounter_date LIKE ?",
                    (f"{d}%",),
                ).fetchall()
                for r in rows:
                    bump(r["chunk_id"], 2.5)

        for med in pq.medications:
            rows = cur.execute(
                "SELECT chunk_id FROM medications WHERE lower(medication) LIKE ?",
                (f"%{med.lower()}%",),
            ).fetchall()
            for r in rows:
                bump(r["chunk_id"], 2.0)

        for code in pq.icd_codes:
            rows = cur.execute(
                "SELECT chunk_id FROM diagnosis_codes WHERE code = ?", (code.upper(),)
            ).fetchall()
            for r in rows:
                bump(r["chunk_id"], 2.5)

        for prov in pq.providers:
            rows = cur.execute(
                "SELECT chunk_id FROM providers WHERE lower(provider) LIKE ?",
                (f"%{prov.lower()}%",),
            ).fetchall()
            for r in rows:
                bump(r["chunk_id"], 2.0)

        # Intent / section filters
        for intent in pq.intent_sections:
            like = f"%{intent.replace('_', ' ').lower()}%"
            rows = cur.execute(
                "SELECT chunk_id FROM chunks WHERE lower(COALESCE(section,'')) LIKE ? OR lower(COALESCE(table_type,'')) LIKE ?",
                (like, f"%{intent.lower()}%"),
            ).fetchall()
            for r in rows:
                bump(r["chunk_id"], 1.0)

            if intent == "signature":
                for r in cur.execute("SELECT chunk_id FROM signatures").fetchall():
                    bump(r["chunk_id"], 1.5)
            if intent == "laboratory_results":
                for r in cur.execute("SELECT chunk_id FROM laboratory_results").fetchall():
                    bump(r["chunk_id"], 0.8)
            if intent in {"medications", "plan"}:
                for r in cur.execute("SELECT chunk_id FROM medications").fetchall():
                    bump(r["chunk_id"], 0.5)
            if intent == "imaging_orders":
                for r in cur.execute("SELECT chunk_id FROM imaging_orders").fetchall():
                    bump(r["chunk_id"], 1.2)
            if intent == "referrals":
                for r in cur.execute("SELECT chunk_id FROM referrals").fetchall():
                    bump(r["chunk_id"], 1.2)

        # HbA1c / lab needle from free text
        ql = pq.raw.lower()
        if "hba1c" in ql:
            for r in cur.execute(
                "SELECT chunk_id FROM laboratory_results WHERE lower(lab) LIKE '%hba1c%'"
            ).fetchall():
                bump(r["chunk_id"], 2.0)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
