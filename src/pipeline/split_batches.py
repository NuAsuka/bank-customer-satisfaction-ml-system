from __future__ import annotations
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/dataset-sots-seti.xlsx")
OUT_DIR = Path("data/incoming")
HISTORY_NAME = "history.csv"    # исходная часть
BATCH_GLOB = "batch_*.csv"      # порции: batch_01.csv, batch_02.csv, …
BASE_SHARE = 0.60               # доля исходной части
N_BATCHES = 4                   # три–пять порций по методичке
DATE_COL = "review_dttm"


def day_boundary(day_values, target: int) -> int:
    """Первый индекс строки с датой строго больше даты строки target."""
    return int((day_values > day_values[target]).argmax())


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"нет исходного файла: {RAW_PATH}")

    df = pd.read_excel(RAW_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    rows_raw = len(df)

    # Устойчивая сортировка: при совпадении отметок времени порядок строк
    # детерминирован и не зависит от окружения.
    df = df.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
    days = df[DATE_COL].dt.normalize().to_numpy()

    # Границы режем по суткам, а не по номерам строк: иначе один день попадёт
    # и в исходную часть, и в порцию — то есть в обучение утечёт будущее.
    cut = day_boundary(days, int(round(rows_raw * BASE_SHARE)))
    history = df.iloc[:cut].copy()
    tail = df.iloc[cut:].copy()
    tail_days = days[cut:]

    edges = [0]
    for i in range(1, N_BATCHES):
        edges.append(day_boundary(tail_days, round(i * len(tail) / N_BATCHES)))
    edges.append(len(tail))

    parts: list[tuple[Path, pd.DataFrame]] = [(OUT_DIR / HISTORY_NAME, history)]
    for i in range(N_BATCHES):
        parts.append(
            (OUT_DIR / f"batch_{i + 1:02d}.csv",
             tail.iloc[edges[i]:edges[i + 1]].copy())
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob(BATCH_GLOB):   # чтобы не осталось порций от прошлой схемы
        stale.unlink()
    for path, part in parts:
        part.to_csv(path, index=False, encoding="utf-8")

    # --- контроль ------------------------------------------------------------
    total = sum(len(part) for _, part in parts)
    assert total == rows_raw, f"потеря строк: {total} != {rows_raw}"
    assert 0.55 <= len(history) / rows_raw <= 0.65, (
        f"исходная часть {len(history) / rows_raw:.1%} вышла за «около 60 %»"
    )
    seen: set = set()
    for path, part in parts:   # ни одна дата не должна встречаться дважды
        part_days = set(part[DATE_COL].dt.date)
        assert not (part_days & seen), (
            f"{path.name}: даты пересекаются с предыдущими файлами "
            f"({sorted(part_days & seen)[:3]})"
        )
        seen |= part_days

    print(f"исходный файл: {RAW_PATH} — {rows_raw} строк × {df.shape[1]} колонок")
    for path, part in parts:
        print(
            f"  {str(path):<30} {len(part):>6} строк "
            f"({len(part) / rows_raw * 100:4.1f}%)  "
            f"{part[DATE_COL].min().date()} … {part[DATE_COL].max().date()}  "
            f"негатив {(part['review_emotion'] == 'Негативный').mean() * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
