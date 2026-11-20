import hashlib
import re
from datetime import date
from pathlib import Path

KINDS = {
    "review": ("etl.review_batches", "batch_review_id"),
    "enrichment": ("etl.review_enrichment_batches", "batch_enrichment_id"),
}


def _meta(kind: str) -> tuple[str, str]:
    """Проверка, что запись из существующего потока
    Возврат таблицы и первичного ключа потока
    """
    if kind not in KINDS:
        raise ValueError(f"неизвестный поток: {kind}")
    return KINDS[kind]


def file_sha256(path) -> str:
    """
    Создание контрольной суммы файла.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lookup(conn, kind: str, path) -> tuple[int, str] | None:
    table, key = _meta(kind)
    with conn.cursor() as cur:
        cur.execute(f"SELECT {key}, status FROM {table} WHERE file_sha256 = %s", (file_sha256(path),))
        row = cur.fetchone()
    return (int(row[0]), str(row[1])) if row else None


def find(conn, kind: str, path) -> int | None:
    """Проверка того, что полученный файл не загружался в хранилище ранее в другом пакете."""
    found = lookup(conn, kind, path)
    return found[0] if found else None


def by_status(conn, kind: str, status: str) -> list[int]:
    table, key = _meta(kind)
    with conn.cursor() as cur:
        cur.execute(f"SELECT {key} FROM {table} WHERE status = %s ORDER BY {key}", (status,))
        return [int(row[0]) for row in cur.fetchall()]


def by_name(conn, kind: str, name: str) -> tuple[int, str] | None:
    table, key = _meta(kind)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {key}, status FROM {table} WHERE file_name = %s ORDER BY {key} DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
    return (int(row[0]), str(row[1])) if row else None


def incoming(directory) -> list[Path]:
    folder = Path(directory)
    if not folder.is_dir():
        raise FileNotFoundError(f"нет каталога {folder}")
    numbered = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = re.fullmatch(r"batch_(\d+)\.csv", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    ordered = []
    history = folder / "history.csv"
    if history.is_file():
        ordered.append(history)
    ordered.extend(path for _, path in sorted(numbered))
    return ordered


def register(conn, kind: str, path) -> int:
    """Регистрация пакета (файла) в базе данных"""
    table, key = _meta(kind)
    sha = file_sha256(path)
    with conn.cursor() as cur:
        cur.execute(f"SELECT {key} FROM {table} WHERE file_sha256 = %s", (sha,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur.execute(
            f"""
            INSERT INTO {table} (file_name, file_sha256, status)
            VALUES (%s, %s, 'received')
            RETURNING {key}
            """,
            (Path(path).name, sha),
        )
        batch_id = int(cur.fetchone()[0])
    conn.commit()
    return batch_id


def set_status(
    conn,
    kind: str,
    batch_id: int,
    status: str,
    rows_loaded: int | None = None,
    period_start: date | None = None,
    period_end: date | None = None,
) -> None:
    """Обновлеие свойств уже зарегестрированных пакетов."""
    table, key = _meta(kind)
    fields = ["status = %s"]
    values: list = [status]
    if rows_loaded is not None:
        fields.append("rows_loaded = %s")
        values.append(rows_loaded)
    if period_start is not None:
        fields.append("period_start = %s")
        values.append(period_start)
    if period_end is not None:
        fields.append("period_end = %s")
        values.append(period_end)
    values.append(batch_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {table} SET {', '.join(fields)} WHERE {key} = %s", values)
    conn.commit()
