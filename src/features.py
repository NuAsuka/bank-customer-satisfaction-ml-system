def build(conn, batch_review_id: int | None = None, batch_enrichment_id: int | None = None) -> int:
    with conn.cursor() as cur:
        if batch_review_id is not None:
            cur.execute(
                """
                INSERT INTO staging.training_set
                    (id_client, review_dttm, batch_review_id, batch_enrichment_id,
                     review_source, review_text, product, review_emotion)
                SELECT c.id_client, c.review_dttm, c.batch_review_id, e.batch_enrichment_id,
                       c.review_source, c.review_text, e.product, e.review_emotion
                  FROM staging.review_clean c
                  LEFT JOIN staging.review_enrichment_clean e
                         ON e.id_client = c.id_client AND e.review_dttm = c.review_dttm
                 WHERE c.batch_review_id = %s
                ON CONFLICT (id_client, review_dttm) DO UPDATE
                   SET batch_review_id = EXCLUDED.batch_review_id,
                       batch_enrichment_id = EXCLUDED.batch_enrichment_id,
                       review_source = EXCLUDED.review_source,
                       review_text = EXCLUDED.review_text,
                       product = EXCLUDED.product,
                       review_emotion = EXCLUDED.review_emotion
                """,
                (batch_review_id,),
            )
        elif batch_enrichment_id is not None:
            cur.execute(
                """
                UPDATE staging.training_set t
                   SET product = e.product,
                       review_emotion = e.review_emotion,
                       batch_enrichment_id = e.batch_enrichment_id
                  FROM staging.review_enrichment_clean e
                 WHERE e.batch_enrichment_id = %s
                   AND t.id_client = e.id_client
                   AND t.review_dttm = e.review_dttm
                """,
                (batch_enrichment_id,),
            )
        else:
            cur.execute("DELETE FROM staging.training_set")
            cur.execute(
                """
                INSERT INTO staging.training_set
                    (id_client, review_dttm, batch_review_id, batch_enrichment_id,
                     review_source, review_text, product, review_emotion)
                SELECT c.id_client, c.review_dttm, c.batch_review_id, e.batch_enrichment_id,
                       c.review_source, c.review_text, e.product, e.review_emotion
                  FROM staging.review_clean c
                  LEFT JOIN staging.review_enrichment_clean e
                         ON e.id_client = c.id_client AND e.review_dttm = c.review_dttm
                ON CONFLICT (id_client, review_dttm) DO UPDATE
                   SET batch_review_id = EXCLUDED.batch_review_id,
                       batch_enrichment_id = EXCLUDED.batch_enrichment_id,
                       review_source = EXCLUDED.review_source,
                       review_text = EXCLUDED.review_text,
                       product = EXCLUDED.product,
                       review_emotion = EXCLUDED.review_emotion
                """
            )
        affected = cur.rowcount
    conn.commit()
    return affected
