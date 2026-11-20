-- =====================================================================
-- Хранилище системы анализа удовлетворённости клиентов банка.
-- Схемы: etl (реестры загрузок и журнал запусков), raw (данные в том
-- виде, в каком пришли от источника), quarantine (маркеры отбраковки),
-- staging (очищенные данные, прогнозы, обучающая выборка).
-- =====================================================================

CREATE DATABASE airflow;

CREATE SCHEMA IF NOT EXISTS etl;
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS quarantine;
CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE etl.review_batches (
    batch_review_id bigint      GENERATED ALWAYS AS IDENTITY,
    file_name       text        NOT NULL,
    file_sha256     char(64)    NOT NULL,
    rows_loaded     integer,
    period_start    date,
    period_end      date,
    loaded_at       timestamptz NOT NULL DEFAULT now(),
    status          text        NOT NULL DEFAULT 'received',
    CONSTRAINT pk_review_batches PRIMARY KEY (batch_review_id),
    CONSTRAINT ux_review_batches_sha256 UNIQUE (file_sha256),
    CONSTRAINT ck_review_batches_status
        CHECK (status IN ('received', 'processing', 'done', 'blocked', 'failed')),
    CONSTRAINT ck_review_batches_period
        CHECK (period_start IS NULL OR period_end IS NULL OR period_start <= period_end),
    CONSTRAINT ck_review_batches_rows
        CHECK (rows_loaded IS NULL OR rows_loaded >= 0)
);

-- Реестр загруженных файлов разметки.
CREATE TABLE etl.review_enrichment_batches (
    batch_enrichment_id bigint      GENERATED ALWAYS AS IDENTITY,
    file_name           text        NOT NULL,
    file_sha256         char(64)    NOT NULL,
    rows_loaded         integer,
    period_start        date,
    period_end          date,
    loaded_at           timestamptz NOT NULL DEFAULT now(),
    status              text        NOT NULL DEFAULT 'received',
    CONSTRAINT pk_review_enrichment_batches PRIMARY KEY (batch_enrichment_id),
    CONSTRAINT ux_review_enrichment_batches_sha256 UNIQUE (file_sha256),
    CONSTRAINT ck_review_enrichment_batches_status
        CHECK (status IN ('received', 'processing', 'done', 'blocked', 'failed')),
    CONSTRAINT ck_review_enrichment_batches_period
        CHECK (period_start IS NULL OR period_end IS NULL OR period_start <= period_end),
    CONSTRAINT ck_review_enrichment_batches_rows
        CHECK (rows_loaded IS NULL OR rows_loaded >= 0)
);

-- Журнал запусков
CREATE TABLE etl.run_log (
    run_id              bigint      GENERATED ALWAYS AS IDENTITY,
    batch_review_id     bigint,
    batch_enrichment_id bigint,
    stage               text        NOT NULL,
    status              text        NOT NULL,
    attempt             smallint    NOT NULL DEFAULT 1,
    started_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz,
    rows_in             integer,
    rows_out            integer,
    rows_rejected       integer,
    model_version       text,
    message             text,
    CONSTRAINT pk_run_log PRIMARY KEY (run_id),
    CONSTRAINT fk_run_log_review_batch
        FOREIGN KEY (batch_review_id) REFERENCES etl.review_batches (batch_review_id),
    CONSTRAINT fk_run_log_enrichment_batch
        FOREIGN KEY (batch_enrichment_id) REFERENCES etl.review_enrichment_batches (batch_enrichment_id),
    CONSTRAINT ck_run_log_stage
        CHECK (stage IN ('load', 'quality', 'features', 'scoring', 'train', 'kpi')),
    CONSTRAINT ck_run_log_status
        CHECK (status IN ('running', 'success', 'failed', 'blocked', 'skipped')),
    CONSTRAINT ck_run_log_stream
        CHECK (num_nonnulls(batch_review_id, batch_enrichment_id) <= 1),
    CONSTRAINT ck_run_log_attempt CHECK (attempt >= 1),
    CONSTRAINT ck_run_log_finish
        CHECK (finished_at IS NULL OR finished_at >= started_at)
);

-- raw

CREATE TABLE raw.review (
    id_client       varchar(32) NOT NULL,
    review_dttm     timestamptz NOT NULL,
    batch_review_id bigint      NOT NULL,
    review_source   varchar(50),
    review_text     text,
    received_at     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT pk_review PRIMARY KEY (id_client, review_dttm),
    CONSTRAINT fk_review_batch
        FOREIGN KEY (batch_review_id) REFERENCES etl.review_batches (batch_review_id)
);

-- Партия разметки
CREATE TABLE raw.review_enrichment (
    id_client           varchar(32) NOT NULL,
    review_dttm         timestamptz NOT NULL,
    batch_enrichment_id bigint      NOT NULL,
    product             varchar(50),
    review_emotion      varchar(20),
    received_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT pk_review_enrichment PRIMARY KEY (id_client, review_dttm),
    CONSTRAINT fk_review_enrichment_batch
        FOREIGN KEY (batch_enrichment_id)
        REFERENCES etl.review_enrichment_batches (batch_enrichment_id)
);

-- quarantine

-- Маркер отбраковки отзыва: ссылка на строку raw.review и причина.
CREATE TABLE quarantine.review (
    id_client       varchar(32) NOT NULL,
    review_dttm     timestamptz NOT NULL,
    batch_review_id bigint      NOT NULL,
    reason_text     text,
    quarantine_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT pk_quarantine_review PRIMARY KEY (id_client, review_dttm),
    CONSTRAINT fk_quarantine_review_raw
        FOREIGN KEY (id_client, review_dttm) REFERENCES raw.review (id_client, review_dttm),
    CONSTRAINT fk_quarantine_review_batch
        FOREIGN KEY (batch_review_id) REFERENCES etl.review_batches (batch_review_id)
);

-- Маркер отбраковки строки разметки.
CREATE TABLE quarantine.review_enrichment (
    id_client           varchar(32) NOT NULL,
    review_dttm         timestamptz NOT NULL,
    batch_enrichment_id bigint      NOT NULL,
    reason_text         text,
    quarantine_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT pk_quarantine_review_enrichment PRIMARY KEY (id_client, review_dttm),
    CONSTRAINT fk_quarantine_review_enrichment_raw
        FOREIGN KEY (id_client, review_dttm)
        REFERENCES raw.review_enrichment (id_client, review_dttm),
    CONSTRAINT fk_quarantine_review_enrichment_batch
        FOREIGN KEY (batch_enrichment_id)
        REFERENCES etl.review_enrichment_batches (batch_enrichment_id)
);

-- ------------------------------------------------------------ staging

-- Отзывы, прошедшие проверки качества
CREATE TABLE staging.review_clean (
    id_client       varchar(32) NOT NULL,
    review_dttm     timestamptz NOT NULL,
    batch_review_id bigint      NOT NULL,
    review_source   varchar(50) NOT NULL,
    review_text     text        NOT NULL,
    CONSTRAINT pk_review_clean PRIMARY KEY (id_client, review_dttm),
    CONSTRAINT fk_review_clean_raw
        FOREIGN KEY (id_client, review_dttm) REFERENCES raw.review (id_client, review_dttm),
    CONSTRAINT fk_review_clean_batch
        FOREIGN KEY (batch_review_id) REFERENCES etl.review_batches (batch_review_id),
    CONSTRAINT ck_review_clean_text
        CHECK (char_length(btrim(review_text)) >= 20)
);

-- Разметка, прошедшая проверки качества.
CREATE TABLE staging.review_enrichment_clean (
    id_client           varchar(32) NOT NULL,
    review_dttm         timestamptz NOT NULL,
    batch_enrichment_id bigint      NOT NULL,
    product             varchar(50) NOT NULL,
    review_emotion      varchar(20) NOT NULL,
    CONSTRAINT pk_review_enrichment_clean PRIMARY KEY (id_client, review_dttm),
    CONSTRAINT fk_review_enrichment_clean_raw
        FOREIGN KEY (id_client, review_dttm)
        REFERENCES raw.review_enrichment (id_client, review_dttm),
    CONSTRAINT fk_review_enrichment_clean_batch
        FOREIGN KEY (batch_enrichment_id)
        REFERENCES etl.review_enrichment_batches (batch_enrichment_id),
    CONSTRAINT ck_review_enrichment_clean_emotion
        CHECK (review_emotion IN ('Позитивный', 'Негативный'))
);

-- Предсказания по версиям модели
CREATE TABLE staging.ml_predictions (
    id_client              varchar(32) NOT NULL,
    review_dttm            timestamptz NOT NULL,
    model_version          varchar(50) NOT NULL,
    batch_review_id        bigint      NOT NULL,
    review_emotion_prob    numeric(6,5),
    review_emotion_predict varchar(20),
    scored_at              timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT pk_ml_predictions PRIMARY KEY (id_client, review_dttm, model_version),
    CONSTRAINT fk_ml_predictions_clean
        FOREIGN KEY (id_client, review_dttm)
        REFERENCES staging.review_clean (id_client, review_dttm),
    CONSTRAINT fk_ml_predictions_batch
        FOREIGN KEY (batch_review_id) REFERENCES etl.review_batches (batch_review_id),
    CONSTRAINT ck_ml_predictions_prob
        CHECK (review_emotion_prob IS NULL OR (review_emotion_prob >= 0 AND review_emotion_prob <= 1)),
    CONSTRAINT ck_ml_predictions_label
        CHECK (review_emotion_predict IS NULL
               OR review_emotion_predict IN ('Позитивный', 'Негативный'))
);

-- Обучающая выборка: очищенные отзывы плюс пришедшая разметка.
CREATE TABLE staging.training_set (
    id_client           varchar(32) NOT NULL,
    review_dttm         timestamptz NOT NULL,
    batch_review_id     bigint      NOT NULL,
    batch_enrichment_id bigint,
    review_source       varchar(50) NOT NULL,
    review_text         text        NOT NULL,
    product             varchar(50),
    review_emotion      varchar(20),
    CONSTRAINT pk_training_set PRIMARY KEY (id_client, review_dttm),
    CONSTRAINT fk_training_set_clean
        FOREIGN KEY (id_client, review_dttm)
        REFERENCES staging.review_clean (id_client, review_dttm),
    CONSTRAINT fk_training_set_review_batch
        FOREIGN KEY (batch_review_id) REFERENCES etl.review_batches (batch_review_id),
    CONSTRAINT fk_training_set_enrichment_batch
        FOREIGN KEY (batch_enrichment_id)
        REFERENCES etl.review_enrichment_batches (batch_enrichment_id),
    CONSTRAINT ck_training_set_emotion
        CHECK (review_emotion IS NULL OR review_emotion IN ('Позитивный', 'Негативный'))
);

-- индексы

-- описание

COMMENT ON TABLE etl.review_batches IS 'Реестр загруженных файлов отзывов';
COMMENT ON TABLE etl.review_enrichment_batches IS 'Реестр загруженных файлов разметки';
COMMENT ON TABLE etl.run_log IS 'Журнал запусков';
COMMENT ON TABLE raw.review IS 'Порция отзывов в том виде, в каком пришла от источника';
COMMENT ON TABLE raw.review_enrichment IS 'Разметка к отзывам';
COMMENT ON TABLE quarantine.review IS 'Реестр отбракованных отзывов из raw.review';
COMMENT ON TABLE quarantine.review_enrichment IS 'Маркер отбраковки строки разметки';
COMMENT ON TABLE staging.review_clean IS 'Отзывы, прошедшие проверки качества';
COMMENT ON TABLE staging.review_enrichment_clean IS 'Разметка, прошедшая проверки качества';
COMMENT ON TABLE staging.ml_predictions IS 'Предсказания по версиям модели';
COMMENT ON TABLE staging.training_set IS 'Обучающая выборка: отзывы плюс разметка';

COMMENT ON COLUMN etl.run_log.stage IS 'load, quality, features, scoring, train, kpi';
COMMENT ON COLUMN etl.run_log.status IS 'running, success, failed, blocked, skipped';
COMMENT ON COLUMN etl.review_batches.status IS 'received, processing, done, blocked, failed';
COMMENT ON COLUMN etl.review_enrichment_batches.status IS 'received, processing, done, blocked, failed';
