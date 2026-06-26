# tests/test_logger.py
import logging
from logger import PipelineLogger

def test_step_logs_step_number_name_and_model(caplog):
    logger = PipelineLogger()
    with caplog.at_level(logging.INFO, logger="pipeline"):
        logger.step(1, 3, "Transcribing audio", "whisper-1")
    assert "STEP 1/3" in caplog.text
    assert "Transcribing audio" in caplog.text
    assert "whisper-1" in caplog.text

def test_done_logs_checkmark_and_message(caplog):
    logger = PipelineLogger()
    with caplog.at_level(logging.INFO, logger="pipeline"):
        logger.done("Transcript ready")
    assert "✓" in caplog.text
    assert "Transcript ready" in caplog.text

def test_error_logs_cross_step_and_error(caplog):
    logger = PipelineLogger()
    with caplog.at_level(logging.ERROR, logger="pipeline"):
        logger.error("Transcription", "API timeout")
    assert "✗" in caplog.text
    assert "Transcription" in caplog.text
    assert "API timeout" in caplog.text

def test_finish_logs_done_and_duration(caplog):
    logger = PipelineLogger()
    with caplog.at_level(logging.INFO, logger="pipeline"):
        logger.finish(11.4)
    assert "DONE" in caplog.text
    assert "11.4" in caplog.text
