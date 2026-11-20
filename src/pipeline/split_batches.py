from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/dataset-sots-seti.xlsx")
OUT_DIR = Path("data/incoming")
FEATURES_DIR = OUT_DIR / "features"
LABELS_DIR = OUT_DIR / "labels"
HISTORY_NAME = "history.csv"
BATCH_GLOB = "batch_*.csv"
BASE_SHARE = 0.60
N_BATCHES = 4
DATE_COL = "review_dttm"
FEATURE_COLUMNS = ["id_client", "review_dttm", "review_source", "review_text"]
LABEL_COLUMNS = ["id_client", "review_dttm", "product", "review_emotion"]
HISTORY_COLUMNS = ["id_client", "review_dttm", "review_source", "review_text", "product", "review_emotion"]


def day_boundary(day_values, target: int) -> int:
    return int((day_values > day_values[target]).argmax())


def split_frame(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    df = df.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
    days = df[DATE_COL].dt.normalize().to_numpy()
    rows_raw = len(df)

    cut = day_boundary(days, int(round(rows_raw * BASE_SHARE)))
    history = df.iloc[:cut].copy()
    tail = df.iloc[cut:].copy()
    tail_days = days[cut:]

    edges = [0]
    for i in range(1, N_BATCHES):
        edges.append(day_boundary(tail_days, round(i * len(tail) / N_BATCHES)))
    edges.append(len(tail))

    parts = [(HISTORY_NAME, history)]
    for i in range(N_BATCHES):
        parts.append((f"batch_{i + 1:02d}.csv", tail.iloc[edges[i]:edges[i + 1]].copy()))
    return parts


def as_text(part: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(part.to_csv(index=False)), dtype=str, keep_default_na=False
    )


def write_parts(parts: list[tuple[str, pd.DataFrame]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob(BATCH_GLOB):
        stale.unlink()

    for folder in (FEATURES_DIR, LABELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.csv"):
            stale.unlink()

    for name, part in parts:
        frame = as_text(part)
        if name == HISTORY_NAME:
            frame[HISTORY_COLUMNS].to_csv(OUT_DIR / name, index=False, encoding="utf-8")
        frame[FEATURE_COLUMNS].to_csv(FEATURES_DIR / name, index=False, encoding="utf-8")
        frame[LABEL_COLUMNS].to_csv(LABELS_DIR / name, index=False, encoding="utf-8")


def check(parts: list[tuple[str, pd.DataFrame]], rows_raw: int) -> None:
    total = sum(len(part) for _, part in parts)
    assert total == rows_raw, f"потеря строк: {total} != {rows_raw}"
    assert 0.55 <= len(parts[0][1]) / rows_raw <= 0.65, (
        f"исходная часть {len(parts[0][1]) / rows_raw:.1%} вышла за «около 60 %»"
    )
    seen: set = set()
    for name, part in parts:
        part_days = set(part[DATE_COL].dt.date)
        assert not (part_days & seen), (
            f"{name}: даты пересекаются с предыдущими файлами ({sorted(part_days & seen)[:3]})"
        )
        seen |= part_days


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"нет исходного файла: {RAW_PATH}")

    df = pd.read_excel(RAW_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    rows_raw = len(df)
    missing = [column for column in FEATURE_COLUMNS + LABEL_COLUMNS if column not in df.columns]
    if missing:
        raise SystemExit(f"в выгрузке нет колонок: {missing}")

    parts = split_frame(df)
    check(parts, rows_raw)
    write_parts(parts)

    print(f"исходный файл: {RAW_PATH} — {rows_raw} строк × {df.shape[1]} колонок")
    for name, part in parts:
        print(
            f"  {name:<14} {len(part):>6} строк ({len(part) / rows_raw * 100:4.1f}%)  "
            f"{part[DATE_COL].min().date()} … {part[DATE_COL].max().date()}  "
            f"негатив {(part['review_emotion'] == 'Негативный').mean() * 100:.2f}%"
        )
    print(f"базовая часть: {OUT_DIR / HISTORY_NAME}")
    print(f"признаки: {FEATURES_DIR}")
    print(f"разметка: {LABELS_DIR}")


if __name__ == "__main__":
    main()
