-- ============================================================================
-- Нормализованная ER-модель датасета отзывов (history.csv).
-- PostgreSQL 16, схема staging.
--
-- Модель приведена к 3НФ: справочные сущности вынесены в отдельные таблицы,
-- иерархии company -> business_line -> product и review_theme -> review_subtheme
-- заданы через внешние ключи. Это не размерная модель Кимбалла: сущность
-- review хранит сам отзыв, а не «факт» с surrogate-измерениями.
--
--   company 1─N business_line 1─N product
--   review_theme 1─N review_subtheme
--   gender / education_level / marital_status / country /
--   client_segment / age_segment 1─N client
--   client 1─N review N─1 product, review_source, review_reason,
--                                review_subtheme, review_emotion,
--                                review_mark, review_solution
--
-- Гранулярность review: одна строка = один отзыв.
-- Скрипт идемпотентен: повторный запуск не удаляет и не пересоздаёт данные.
-- ============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS staging;
COMMENT ON SCHEMA staging IS 'Очищенные данные — источник правды (нормализованная ER-модель)';

-- Справочники клиентских атрибутов ------------------------------------------

CREATE TABLE IF NOT EXISTS staging.gender (
    gender_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    gender_code text NOT NULL UNIQUE,
    gender_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.education_level (
    education_level_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    education_level_code text NOT NULL UNIQUE,
    education_level_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.marital_status (
    marital_status_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    marital_status_code text NOT NULL UNIQUE,
    marital_status_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.country (
    country_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    country_code text NOT NULL UNIQUE,
    country_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.client_segment (
    client_segment_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    client_segment_code text NOT NULL UNIQUE,
    client_segment_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.age_segment (
    age_segment_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    age_segment_code text NOT NULL UNIQUE,
    age_segment_name text NOT NULL
);

-- Иерархия «компания → бизнес-линия → продукт» ------------------------------

CREATE TABLE IF NOT EXISTS staging.company (
    company_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    company_code text NOT NULL UNIQUE,
    company_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.business_line (
    business_line_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    business_line_code text     NOT NULL,
    business_line_name text     NOT NULL,
    company_id         smallint NOT NULL REFERENCES staging.company (company_id),
    CONSTRAINT business_line_uq UNIQUE (company_id, business_line_code)
);

CREATE TABLE IF NOT EXISTS staging.product (
    product_id       smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_code     text     NOT NULL,
    product_name     text     NOT NULL,
    business_line_id smallint NOT NULL REFERENCES staging.business_line (business_line_id),
    CONSTRAINT product_uq UNIQUE (business_line_id, product_code)
);

-- Иерархия «тема → подтема» -------------------------------------------------

CREATE TABLE IF NOT EXISTS staging.review_theme (
    theme_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    theme_code text NOT NULL UNIQUE,
    theme_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.review_subtheme (
    subtheme_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    subtheme_code text     NOT NULL,
    subtheme_name text     NOT NULL,
    theme_id      smallint NOT NULL REFERENCES staging.review_theme (theme_id),
    CONSTRAINT review_subtheme_uq UNIQUE (theme_id, subtheme_code)
);

-- Справочники отзыва ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS staging.review_source (
    source_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_code text NOT NULL UNIQUE,
    source_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.review_emotion (
    emotion_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    emotion_code text NOT NULL UNIQUE,
    emotion_name text NOT NULL
);

COMMENT ON TABLE staging.review_emotion IS 'Эмоциональная окраска отзыва; целевая переменная модели';

CREATE TABLE IF NOT EXISTS staging.review_mark (
    mark_id    smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    mark_code  text     NOT NULL UNIQUE,
    mark_name  text     NOT NULL,
    mark_value smallint CHECK (mark_value BETWEEN 1 AND 5)
);

COMMENT ON COLUMN staging.review_mark.mark_value IS
    'Числовая оценка 1–5; NULL для значений вида «без оценки»';

CREATE TABLE IF NOT EXISTS staging.review_reason (
    reason_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    reason_code text NOT NULL UNIQUE,
    reason_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.review_solution (
    solution_id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    solution_code text NOT NULL UNIQUE,
    solution_name text NOT NULL
);

-- Клиент --------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS staging.client (
    client_id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    client_external_id          text     NOT NULL UNIQUE,
    gender_id                   smallint REFERENCES staging.gender (gender_id),
    education_level_id          smallint REFERENCES staging.education_level (education_level_id),
    marital_status_id           smallint REFERENCES staging.marital_status (marital_status_id),
    children_cnt                smallint CHECK (children_cnt >= 0),
    country_id                  smallint REFERENCES staging.country (country_id),
    client_segment_id           smallint REFERENCES staging.client_segment (client_segment_id),
    subscription_important_flg  boolean,
    new_flg                     boolean,
    influencer_flg              boolean,
    age_segment_id              smallint REFERENCES staging.age_segment (age_segment_id),
    first_review_dttm           timestamptz NOT NULL
);

COMMENT ON TABLE staging.client IS
    'Клиент; атрибуты взяты из первого по времени отзыва клиента в датасете';
COMMENT ON COLUMN staging.client.client_external_id IS
    'Идентификатор клиента из источника (натуральный ключ)';

-- Отзыв ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS staging.review (
    review_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    client_id    bigint      NOT NULL REFERENCES staging.client (client_id),
    review_dttm  timestamptz NOT NULL,
    finish_dttm  timestamptz,
    product_id   smallint    NOT NULL REFERENCES staging.product (product_id),
    source_id    smallint    NOT NULL REFERENCES staging.review_source (source_id),
    reason_id    smallint    NOT NULL REFERENCES staging.review_reason (reason_id),
    subtheme_id  smallint    NOT NULL REFERENCES staging.review_subtheme (subtheme_id),
    emotion_id   smallint    REFERENCES staging.review_emotion (emotion_id),
    mark_id      smallint    REFERENCES staging.review_mark (mark_id),
    solution_id  smallint    REFERENCES staging.review_solution (solution_id),
    review_text  text,
    csat_score   smallint CHECK (csat_score BETWEEN 1 AND 5),
    CONSTRAINT review_natural_key UNIQUE (client_id, review_dttm, source_id)
);

COMMENT ON TABLE staging.review IS 'Отзыв: одна строка = один отзыв клиента';
COMMENT ON COLUMN staging.review.emotion_id IS
    'Эмоциональная окраска (таргет); NULL допустим — разметка приходит с лагом';

CREATE INDEX IF NOT EXISTS review_dttm_idx    ON staging.review (review_dttm);
CREATE INDEX IF NOT EXISTS review_client_idx  ON staging.review (client_id);
CREATE INDEX IF NOT EXISTS review_product_idx ON staging.review (product_id);
CREATE INDEX IF NOT EXISTS review_source_idx  ON staging.review (source_id);
CREATE INDEX IF NOT EXISTS review_emotion_idx ON staging.review (emotion_id);
CREATE INDEX IF NOT EXISTS review_subtheme_idx ON staging.review (subtheme_id);

COMMIT;
