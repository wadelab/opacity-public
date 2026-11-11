import datetime as dt

from opacity import offpeak

UTC = dt.timezone.utc


def at(day: int, h: int, m: int = 0) -> dt.datetime:
    return dt.datetime(2026, 10, day, h, m, tzinfo=UTC)  # 26 Oct 2026 is a Monday


def test_peak_windows_on_weekdays_with_margin():
    assert offpeak.is_peak(at(26, 2)) and offpeak.is_peak(at(26, 9, 59))
    assert not offpeak.is_peak(at(26, 0, 30)) and not offpeak.is_peak(at(26, 4, 0)) and not offpeak.is_peak(at(26, 12))
    assert offpeak.is_peak(at(26, 0, 50))           # inside the 15-minute margin before 01:00
    assert not offpeak.is_peak(at(26, 5, 30)) and offpeak.is_peak(at(26, 5, 50))


def test_weekends_are_offpeak():
    assert not offpeak.is_peak(at(24, 2)) and not offpeak.is_peak(at(25, 8))  # Saturday, Sunday


def test_next_offpeak():
    assert offpeak.next_offpeak(at(26, 7)) == at(26, 10)
    assert offpeak.next_offpeak(at(26, 2)) == at(26, 4)
    assert offpeak.next_offpeak(at(26, 12)) == at(26, 12)


def test_override_skips_the_wait(monkeypatch):
    monkeypatch.setenv("OFFPEAK_OVERRIDE", "1")
    seen = []
    offpeak.wait(seen.append)
    assert seen and "not waiting" in seen[0]
