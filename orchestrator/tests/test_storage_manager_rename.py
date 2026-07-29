import json
import os
import tempfile

from managers.storage_manager import StorageManager


def _storage(tmp_root):
    return StorageManager(session_name="Session_001", output_root=tmp_root)


def test_event_dir_defaults_to_event_id_before_any_rename():
    with tempfile.TemporaryDirectory() as tmp:
        storage = _storage(tmp)
        path = storage.event_dir("EVT_000001")
        assert os.path.basename(path) == "EVT_000001"
        assert os.path.basename(os.path.dirname(path)) == "Session_001"


def test_rename_event_for_coach_number_moves_existing_folder_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        storage = _storage(tmp)
        old_path = storage.event_dir("EVT_000001")
        os.makedirs(old_path)
        with open(os.path.join(old_path, "marker.txt"), "w") as f:
            f.write("hello")

        storage.rename_event_for_coach_number("EVT_000001", "12345")
        new_path = storage.event_dir("EVT_000001")

        assert os.path.basename(new_path) == "12345"
        assert not os.path.isdir(old_path)
        assert os.path.exists(os.path.join(new_path, "marker.txt"))


def test_rename_event_collision_gets_deduped_not_overwritten():
    with tempfile.TemporaryDirectory() as tmp:
        storage = _storage(tmp)

        path1 = storage.event_dir("EVT_000001")
        os.makedirs(path1)
        storage.rename_event_for_coach_number("EVT_000001", "12345")

        path2 = storage.event_dir("EVT_000002")
        os.makedirs(path2)
        storage.rename_event_for_coach_number("EVT_000002", "12345")  # same number, different coach

        final1 = storage.event_dir("EVT_000001")
        final2 = storage.event_dir("EVT_000002")
        assert final1 != final2
        assert os.path.isdir(final1)
        assert os.path.isdir(final2)


def test_rename_session_for_train_number_only_applies_once():
    with tempfile.TemporaryDirectory() as tmp:
        storage = _storage(tmp)
        storage.rename_session_for_train_number("11111")
        storage.rename_session_for_train_number("99999")  # second call must be ignored

        assert os.path.isdir(os.path.join(tmp, "11111"))
        assert not os.path.isdir(os.path.join(tmp, "99999"))


def test_rename_session_relocates_already_created_event_folders():
    with tempfile.TemporaryDirectory() as tmp:
        storage = _storage(tmp)
        path = storage.event_dir("EVT_000001")
        os.makedirs(path)

        storage.rename_session_for_train_number("11111")
        new_path = storage.event_dir("EVT_000001")

        assert os.path.isdir(new_path)
        assert "11111" in new_path
        assert os.path.basename(new_path) == "EVT_000001"


def test_event_id_survives_inside_event_json_after_rename():
    from core.models import EventStatus, EventWindow

    with tempfile.TemporaryDirectory() as tmp:
        storage = _storage(tmp)
        event = EventWindow(event_id="EVT_000001", coach_index=1, start_ts_ms=0,
                             end_ts_ms=1000, status=EventStatus.BOUNDED, coach_number="12345")
        from managers.frame_assignment_manager import EventFrames
        storage.save_event(event, EventFrames(event_id="EVT_000001"), [], [], 0.1)
        storage.rename_event_for_coach_number("EVT_000001", "12345")

        new_path = storage.event_dir("EVT_000001")
        with open(os.path.join(new_path, "event.json")) as f:
            payload = json.load(f)
        assert payload["event_id"] == "EVT_000001"   # identity preserved despite folder rename
