-- ============================================================================
-- Демонстрационные аналитические запросы к ER-модели staging.
--
-- Запросы показывают корректную работу связей между сущностями:
-- соединение нескольких таблиц, агрегаты, фильтрацию, группировку,
-- CTE и оконные функции.
--
-- Выполнение:
--   psql postgresql://admin:1@localhost:5433/database \
--        -f src/sql/02_analytics_queries.sql
-- ============================================================================


-- ============================================================================
-- Запрос 1. Негатив по продуктам и площадкам
--
-- Для каждого сочетания «компания → бизнес-линия → продукт → площадка»
-- считаются:
--   * число отзывов с текстом;
--   * число негативных и неразмеченных отзывов;
--   * доля негатива;
--   * средний CSAT и средняя выставленная оценка.
--
-- Фильтры:
--   * период истории — первое полугодие 2025 года;
--   * только отзывы с непустым текстом (именно их обрабатывает модель);
--   * только группы с достаточным объёмом наблюдений (не менее 30 отзывов).
--
-- Соединяются: review, product, business_line, company, review_source,
-- review_emotion, review_mark.
-- ============================================================================

WITH product_source_stats AS (
    SELECT
        co.company_name,
        bl.business_line_name,
        p.product_name,
        s.source_name,
        count(*) AS reviews_cnt,
        count(*) FILTER (
            WHERE e.emotion_name = 'Негативный'
        ) AS negative_cnt,
        count(*) FILTER (
            WHERE e.emotion_id IS NULL
        ) AS unlabeled_cnt,
        round(avg(r.csat_score)::numeric, 2) AS avg_csat,
        round(avg(m.mark_value)::numeric, 2) AS avg_mark
    FROM staging.review AS r
    JOIN staging.product AS p
        ON p.product_id = r.product_id
    JOIN staging.business_line AS bl
        ON bl.business_line_id = p.business_line_id
    JOIN staging.company AS co
        ON co.company_id = bl.company_id
    JOIN staging.review_source AS s
        ON s.source_id = r.source_id
    LEFT JOIN staging.review_emotion AS e
        ON e.emotion_id = r.emotion_id
    LEFT JOIN staging.review_mark AS m
        ON m.mark_id = r.mark_id
    WHERE r.review_dttm >= DATE '2025-01-01'
      AND r.review_dttm <  DATE '2025-07-01'
      AND r.review_text IS NOT NULL
      AND btrim(r.review_text) <> ''
    GROUP BY
        co.company_name,
        bl.business_line_name,
        p.product_name,
        s.source_name
)
SELECT
    company_name,
    business_line_name,
    product_name,
    source_name,
    reviews_cnt,
    negative_cnt,
    unlabeled_cnt,
    round(100.0 * negative_cnt / NULLIF(reviews_cnt, 0), 2) AS negative_share_pct,
    avg_csat,
    avg_mark
FROM product_source_stats
WHERE reviews_cnt >= 30
ORDER BY
    negative_share_pct DESC,
    reviews_cnt DESC,
    company_name,
    business_line_name,
    product_name,
    source_name;


-- ============================================================================
-- Запрос 2. Ежемесячная динамика негатива по площадкам и клиентским сегментам
--
-- Для каждого месяца, площадки и сегмента клиента считаются:
--   * число отзывов и число негативных отзывов;
--   * доля негатива;
--   * изменение доли негатива к предыдущему месяцу (в процентных пунктах);
--   * место площадки по доле негатива внутри месяца.
--
-- Дополнительно показывается доля клиентов с известным сегментом,
-- чтобы было видно влияние пропусков в клиентских атрибутах.
--
-- Соединяются: review, review_source, review_emotion, client,
-- client_segment, age_segment.
-- ============================================================================

WITH monthly_source_segment AS (
    SELECT
        date_trunc('month', r.review_dttm)::date AS month,
        s.source_name,
        coalesce(cs.client_segment_name, 'Сегмент не определён') AS client_segment,
        count(*) AS reviews_cnt,
        count(*) FILTER (
            WHERE e.emotion_name = 'Негативный'
        ) AS negative_cnt,
        count(*) FILTER (
            WHERE c.client_segment_id IS NOT NULL
        ) AS reviews_with_known_segment
    FROM staging.review AS r
    JOIN staging.review_source AS s
        ON s.source_id = r.source_id
    LEFT JOIN staging.review_emotion AS e
        ON e.emotion_id = r.emotion_id
    JOIN staging.client AS c
        ON c.client_id = r.client_id
    LEFT JOIN staging.client_segment AS cs
        ON cs.client_segment_id = c.client_segment_id
    WHERE r.review_dttm >= DATE '2025-01-01'
      AND r.review_dttm <  DATE '2025-07-01'
    GROUP BY
        date_trunc('month', r.review_dttm)::date,
        s.source_name,
        coalesce(cs.client_segment_name, 'Сегмент не определён')
),
with_dynamics AS (
    SELECT
        month,
        source_name,
        client_segment,
        reviews_cnt,
        negative_cnt,
        reviews_with_known_segment,
        round(100.0 * negative_cnt / NULLIF(reviews_cnt, 0), 2) AS negative_share_pct,
        round(
            100.0 * reviews_with_known_segment / NULLIF(reviews_cnt, 0), 2
        ) AS known_segment_share_pct,
        lag(100.0 * negative_cnt / NULLIF(reviews_cnt, 0)) OVER (
            PARTITION BY source_name, client_segment
            ORDER BY month
        ) AS prev_month_negative_share_pct
    FROM monthly_source_segment
)
SELECT
    month,
    source_name,
    client_segment,
    reviews_cnt,
    negative_cnt,
    negative_share_pct,
    round(
        negative_share_pct - prev_month_negative_share_pct, 2
    ) AS delta_pp,
    known_segment_share_pct,
    rank() OVER (
        PARTITION BY month
        ORDER BY negative_share_pct DESC, reviews_cnt DESC
    ) AS negative_rank_in_month
FROM with_dynamics
WHERE reviews_cnt >= 20
ORDER BY
    month,
    negative_rank_in_month,
    source_name,
    client_segment;
