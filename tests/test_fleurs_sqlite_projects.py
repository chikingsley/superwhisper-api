from __future__ import annotations

import json
import tarfile
from typing import TYPE_CHECKING, cast

from superwhisper_api.audio.projects.fleurs_sqlite import (
    FleursProject,
    audit_scribe_results,
    connect,
    ensure_run,
    ensure_schema,
    error_profile,
    ingest_samples,
    pending_samples,
    record_result,
    run_transcriptions,
)
from superwhisper_api.audio.projects.persian.cli import (
    PROJECT as PERSIAN_PROJECT,
)
from superwhisper_api.audio.projects.tajikistan.cli import (
    PROJECT as TAJIKISTAN_PROJECT,
)
from superwhisper_api.audio.transcribe import Transcript

if TYPE_CHECKING:
    from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def project(tmp_path: Path) -> FleursProject:
    return FleursProject(
        name="Test",
        dataset="google/fleurs",
        config="fa_ir",
        split="test",
        database=tmp_path / "google_fleurs.sqlite",
        media_dir=tmp_path / "audio",
        language="fas",
        normalize=normalize,
    )


def insert_sample(database: Path, audio_path: Path) -> None:
    conn = connect(database)
    try:
        ensure_schema(conn)
        with conn:
            conn.execute(
                """
                INSERT INTO fleurs_samples (
                    dataset, config, split, hf_id, audio_path, sample_rate, num_samples,
                    ref_text, normalized_ref_text, raw_row_json, created_at, updated_at
                )
                VALUES (
                    'google/fleurs', 'fa_ir', 'test', 'row-1', ?, 16000, 160,
                    'سلام دنیا', 'سلام دنیا', '{}', 'now', 'now'
                )
                """,
                (str(audio_path),),
            )
    finally:
        conn.close()


def insert_empty_ref_sample(database: Path, audio_path: Path) -> None:
    conn = connect(database)
    try:
        ensure_schema(conn)
        with conn:
            conn.execute(
                """
                INSERT INTO fleurs_samples (
                    dataset, config, split, hf_id, audio_path, sample_rate, num_samples,
                    ref_text, normalized_ref_text, raw_row_json, created_at, updated_at
                )
                VALUES (
                    'google/fleurs', 'fa_ir', 'test', 'row-empty-ref', ?, 16000, 160,
                    'MS', '', '{}', 'now', 'now'
                )
                """,
                (str(audio_path),),
            )
    finally:
        conn.close()


def write_fleurs_fixture(
    source_dir: Path,
    config: str,
    split: str,
    rows: list[tuple[str, str, str]],
) -> None:
    data_dir = source_dir / "data" / config
    audio_dir = data_dir / "audio"
    data_dir.mkdir(parents=True)
    audio_dir.mkdir()
    tsv_path = data_dir / f"{split}.tsv"
    with tsv_path.open("w", encoding="utf-8") as handle:
        for row_id, filename, text in rows:
            handle.write(f"{row_id}\t{filename}\t{text}\t{text}\t{text}\t160\tunknown\n")
    with tarfile.open(audio_dir / f"{split}.tar.gz", mode="w:gz") as tar:
        for _, filename, _ in rows:
            audio_path = source_dir / filename
            audio_path.write_bytes(b"placeholder wav")
            tar.add(audio_path, arcname=filename)


def test_ingest_uses_ref_schema_for_both_language_projects(tmp_path: Path) -> None:
    source_dir = tmp_path / "fleurs"
    write_fleurs_fixture(source_dir, "fa_ir", "test", [("fa-1", "fa.wav", "سلام دنیا")])
    write_fleurs_fixture(source_dir, "tg_tj", "test", [("tg-1", "tg.wav", "Салом дунё")])

    for base_project in (PERSIAN_PROJECT, TAJIKISTAN_PROJECT):
        proj = FleursProject(
            name=base_project.name,
            dataset=base_project.dataset,
            config=base_project.config,
            split=base_project.split,
            database=tmp_path / base_project.config / "google_fleurs.sqlite",
            media_dir=tmp_path / base_project.config / "audio",
            language=base_project.language,
            normalize=base_project.normalize,
        )
        stats = ingest_samples(
            proj,
            database=proj.database,
            media_dir=proj.media_dir,
            dataset=proj.dataset,
            config=proj.config,
            split=proj.split,
            limit=1,
            source_dir=source_dir,
        )

        assert stats.added == 1
        conn = connect(proj.database)
        try:
            row = conn.execute(
                """
                SELECT ref_text, normalized_ref_text, audio_path
                FROM fleurs_samples
                """
            ).fetchone()
            assert row[0]
            assert row[1]
            assert (tmp_path / proj.config / "audio" / proj.config / "test").as_posix() in row[2]
        finally:
            conn.close()


def test_error_profile_is_deterministic() -> None:
    assert (
        error_profile(normalized_ref_text="", normalized_pred_text="x", wer=None, cer=None)
        == "empty_ref"
    )
    assert (
        error_profile(normalized_ref_text="x", normalized_pred_text="", wer=1.0, cer=1.0)
        == "empty_pred"
    )
    assert (
        error_profile(normalized_ref_text="abc", normalized_pred_text="abc", wer=0.0, cer=0.0)
        == "exact"
    )
    assert (
        error_profile(normalized_ref_text="abc", normalized_pred_text="xbc", wer=0.2, cer=0.2)
        == "low_wer_high_cer"
    )
    assert (
        error_profile(normalized_ref_text="abc", normalized_pred_text="ab c", wer=0.5, cer=0.05)
        == "high_wer_low_cer"
    )
    assert (
        error_profile(normalized_ref_text="abc", normalized_pred_text="xyz", wer=0.5, cer=0.5)
        == "high_wer_high_cer"
    )
    assert (
        error_profile(normalized_ref_text="abc", normalized_pred_text="abd", wer=0.2, cer=0.05)
        == "low_wer_low_cer"
    )


def test_run_transcriptions_writes_result_and_resumes(tmp_path: Path) -> None:
    proj = project(tmp_path)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"placeholder")
    insert_sample(proj.database, audio)

    def process(path: Path) -> Transcript:
        return Transcript(
            audio_path=str(path),
            provider="elevenlabs",
            model_key="scribe-v2",
            model_id="scribe_v2",
            transcript="سلام دنیا",
            raw_response={"text": "سلام دنیا"},
            recording_id="rec-1",
            duration=1.0,
            processing_time=10,
        )

    stats = run_transcriptions(
        proj,
        database=proj.database,
        dataset=proj.dataset,
        config=proj.config,
        split=proj.split,
        run_name="test-run",
        model="scribe-v2",
        language="fas",
        key=None,
        limit=5,
        max_workers=1,
        process=process,
    )

    assert stats.ok == 1
    assert stats.pending == 0

    conn = connect(proj.database)
    try:
        row = conn.execute(
            """
            SELECT pred_text, normalized_pred_text, wer, cer, error_profile,
                   raw_response_json, error
            FROM transcription_results
            """
        ).fetchone()
        pred_text, normalized, wer, cer, profile, raw_response_json, error = row
        assert pred_text == "سلام دنیا"
        assert normalized == "سلام دنیا"
        assert wer == 0.0
        assert cer == 0.0
        assert profile == "exact"
        assert json.loads(raw_response_json)["raw_response"] == {"text": "سلام دنیا"}
        assert error is None
    finally:
        conn.close()

    resumed = run_transcriptions(
        proj,
        database=proj.database,
        dataset=proj.dataset,
        config=proj.config,
        split=proj.split,
        run_name="test-run",
        model="scribe-v2",
        language="fas",
        key=None,
        limit=5,
        max_workers=1,
        process=process,
    )
    assert resumed.submitted == 0


def test_pending_samples_are_per_run(tmp_path: Path) -> None:
    proj = project(tmp_path)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"placeholder")
    insert_sample(proj.database, audio)
    conn = connect(proj.database)
    try:
        first_run = ensure_run(
            conn,
            run_name="first",
            dataset=proj.dataset,
            config=proj.config,
            split=proj.split,
            model="scribe-v2",
            language="fas",
        )
        second_run = ensure_run(
            conn,
            run_name="second",
            dataset=proj.dataset,
            config=proj.config,
            split=proj.split,
            model="deepgram-nova-3",
            language="fas",
        )
        sample = pending_samples(
            conn,
            run_id=first_run,
            dataset=proj.dataset,
            config=proj.config,
            split=proj.split,
            limit=1,
        )[0]
        record_result(
            conn,
            run_id=first_run,
            sample=sample,
            result=Transcript(
                audio_path=str(audio),
                provider="elevenlabs",
                model_key="scribe-v2",
                model_id="scribe_v2",
                transcript="سلام دنیا",
                raw_response={"text": "سلام دنیا"},
            ),
            normalize=proj.normalize,
        )
        assert (
            pending_samples(
                conn,
                run_id=first_run,
                dataset=proj.dataset,
                config=proj.config,
                split=proj.split,
                limit=1,
            )
            == []
        )
        assert (
            len(
                pending_samples(
                    conn,
                    run_id=second_run,
                    dataset=proj.dataset,
                    config=proj.config,
                    split=proj.split,
                    limit=1,
                )
            )
            == 1
        )
    finally:
        conn.close()


def test_empty_normalized_ref_is_not_submitted(tmp_path: Path) -> None:
    proj = project(tmp_path)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"placeholder")
    insert_empty_ref_sample(proj.database, audio)

    calls = 0

    def process(path: Path) -> Transcript:
        nonlocal calls
        calls += 1
        return Transcript(
            audio_path=str(path),
            provider="elevenlabs",
            model_key="scribe-v2",
            model_id="scribe_v2",
            transcript="MS",
            raw_response={"text": "MS"},
        )

    stats = run_transcriptions(
        proj,
        database=proj.database,
        dataset=proj.dataset,
        config=proj.config,
        split=proj.split,
        run_name="test-run",
        model="scribe-v2",
        language="fas",
        key=None,
        limit=5,
        max_workers=1,
        process=process,
    )

    assert calls == 0
    assert stats.submitted == 0
    assert stats.pending == 0


def test_scribe_audit_uses_deterministic_and_model_paths(tmp_path: Path) -> None:
    proj = project(tmp_path)
    exact_audio = tmp_path / "exact.wav"
    edit_audio = tmp_path / "edit.wav"
    empty_audio = tmp_path / "empty.wav"
    exact_audio.write_bytes(b"placeholder")
    edit_audio.write_bytes(b"placeholder")
    empty_audio.write_bytes(b"placeholder")
    insert_sample(proj.database, exact_audio)
    insert_empty_ref_sample(proj.database, empty_audio)
    conn = connect(proj.database)
    try:
        ensure_schema(conn)
        with conn:
            conn.execute(
                """
                INSERT INTO fleurs_samples (
                    dataset, config, split, hf_id, audio_path, sample_rate, num_samples,
                    ref_text, normalized_ref_text, raw_row_json, created_at, updated_at
                )
                VALUES (
                    'google/fleurs', 'fa_ir', 'test', 'row-edit', ?, 16000, 160,
                    'سلام دنیا', 'سلام دنیا', '{}', 'now', 'now'
                )
                """,
                (str(edit_audio),),
            )
    finally:
        conn.close()

    def process(path: Path) -> Transcript:
        text = "سلام جهان" if path == edit_audio else "سلام دنیا"
        return Transcript(
            audio_path=str(path),
            provider="elevenlabs",
            model_key="scribe-v2",
            model_id="scribe_v2",
            transcript=text,
            raw_response={"text": text},
        )

    run_transcriptions(
        proj,
        database=proj.database,
        dataset=proj.dataset,
        config=proj.config,
        split=proj.split,
        run_name="test-run",
        model="scribe-v2",
        language="fas",
        key=None,
        limit=10,
        max_workers=1,
        process=process,
    )

    calls: list[object] = []
    edit_counts: list[object] = []

    def audit(row: object) -> dict[str, str]:
        assert isinstance(row, dict)
        row_data = cast("dict[object, object]", row)
        calls.append(row)
        edit_counts.append(row_data.get("edit_counts"))
        return {"category": "substitution", "reason": "one word was replaced"}

    stats = audit_scribe_results(
        database=proj.database,
        run_name="test-run",
        model="gpt-5.4-mini",
        limit=0,
        audit=audit,
    )

    assert stats.deterministic == 2
    assert stats.api == 1
    assert stats.pending == 0
    assert len(calls) == 1
    assert edit_counts[0] == {
        "hits": 1,
        "substitutions": 1,
        "insertions": 0,
        "deletions": 0,
    }

    resumed = audit_scribe_results(
        database=proj.database,
        run_name="test-run",
        model="gpt-5.4-mini",
        limit=0,
        audit=audit,
    )
    assert resumed.deterministic == 0
    assert resumed.api == 0

    conn = connect(proj.database)
    try:
        rows = conn.execute(
            """
            SELECT category, source, COUNT(*)
            FROM scribe_audits
            GROUP BY category, source
            ORDER BY category, source
            """
        ).fetchall()
        assert rows == [
            ("exact_match", "deterministic", 1),
            ("skipped_empty_ref", "deterministic", 1),
            ("substitution", "model", 1),
        ]
    finally:
        conn.close()
