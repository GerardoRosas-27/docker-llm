"""Red de memoria asociativa persistente (SQLite) — NO es una red neuronal.

Nodos: task / expert_version / experience.
Enlaces: success / failure_audit / affinity.
Solo experiencias VERIFICADAS (check automático OK o feedback usuario marcado)
entran en recuperación positiva. Fracasos quedan auditados con motivo pero
excluidos de retrieve_good().
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  ref TEXT,
  payload TEXT,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS edges (
  id TEXT PRIMARY KEY,
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  kind TEXT NOT NULL,
  score REAL DEFAULT 0,
  payload TEXT,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS experiences (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  version_id TEXT,
  prompt TEXT,
  output TEXT,
  ok INTEGER NOT NULL,
  source TEXT NOT NULL,  -- 'auto_check' | 'user_feedback'
  reason TEXT,
  created_at REAL
);
CREATE INDEX IF NOT EXISTS ix_exp_task_ok ON experiences(task_id, ok);
CREATE TABLE IF NOT EXISTS improve_log (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  action TEXT NOT NULL,
  accepted INTEGER,
  detail TEXT,
  created_at REAL
);
"""


class MemoryNet:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def _upsert_node(self, kind: str, ref: str, payload: Optional[Dict] = None) -> str:
        nid = f"{kind}:{ref}"
        cur = self._conn.execute("SELECT id FROM nodes WHERE id=?", (nid,))
        if cur.fetchone() is None:
            self._conn.execute(
                "INSERT INTO nodes(id,kind,ref,payload,created_at) VALUES(?,?,?,?,?)",
                (nid, kind, ref, json.dumps(payload or {}), time.time()),
            )
            self._conn.commit()
        return nid

    def record_experience(
        self,
        task_id: str,
        prompt: str,
        output: str,
        ok: bool,
        *,
        version_id: Optional[str] = None,
        source: str = "auto_check",
        reason: str = "",
    ) -> str:
        eid = uuid.uuid4().hex
        self._upsert_node("task", task_id)
        if version_id:
            self._upsert_node("expert_version", version_id)
        self._conn.execute(
            "INSERT INTO experiences(id,task_id,version_id,prompt,output,ok,source,reason,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (eid, task_id, version_id, prompt, output, int(ok), source, reason, time.time()),
        )
        # edges
        if version_id:
            kind = "success" if ok else "failure_audit"
            self._conn.execute(
                "INSERT INTO edges(id,src,dst,kind,score,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, f"expert_version:{version_id}", f"task:{task_id}", kind,
                 1.0 if ok else 0.0, json.dumps({"experience_id": eid, "reason": reason}), time.time()),
            )
        self._conn.commit()
        return eid

    def retrieve_good(self, task_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Solo éxitos verificados (ok=1)."""
        rows = self._conn.execute(
            "SELECT * FROM experiences WHERE task_id=? AND ok=1 ORDER BY created_at DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def failures(self, task_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id,task_id,version_id,reason,source,created_at FROM experiences "
            "WHERE task_id=? AND ok=0 ORDER BY created_at DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        if task_id:
            ok = self._conn.execute("SELECT COUNT(*) c FROM experiences WHERE task_id=? AND ok=1", (task_id,)).fetchone()["c"]
            bad = self._conn.execute("SELECT COUNT(*) c FROM experiences WHERE task_id=? AND ok=0", (task_id,)).fetchone()["c"]
            return {"task_id": task_id, "successes": ok, "failures_audited": bad}
        ok = self._conn.execute("SELECT COUNT(*) c FROM experiences WHERE ok=1").fetchone()["c"]
        bad = self._conn.execute("SELECT COUNT(*) c FROM experiences WHERE ok=0").fetchone()["c"]
        return {"successes": ok, "failures_audited": bad, "n_nodes": self._conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]}

    def log_improve(self, task_id: str, action: str, accepted: Optional[bool], detail: Dict[str, Any]) -> str:
        iid = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO improve_log(id,task_id,action,accepted,detail,created_at) VALUES(?,?,?,?,?,?)",
            (iid, task_id, action, None if accepted is None else int(accepted), json.dumps(detail), time.time()),
        )
        self._conn.commit()
        return iid

    def improve_history(self, task_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if task_id:
            rows = self._conn.execute(
                "SELECT * FROM improve_log WHERE task_id=? ORDER BY created_at DESC LIMIT ?",
                (task_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM improve_log ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d["detail"] or "{}")
            out.append(d)
        return out

    def reset_task(self, task_id: str, *, keep_audit: bool = True) -> Dict[str, int]:
        """Borra éxitos recuperables de una tarea; opcionalmente conserva fallos auditados."""
        if keep_audit:
            cur = self._conn.execute("DELETE FROM experiences WHERE task_id=? AND ok=1", (task_id,))
            n = cur.rowcount
        else:
            cur = self._conn.execute("DELETE FROM experiences WHERE task_id=?", (task_id,))
            n = cur.rowcount
        self._conn.commit()
        return {"deleted": n, "keep_audit": keep_audit}
