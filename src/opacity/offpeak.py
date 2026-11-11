"""Call hosted models off-peak only (Wade, 24 October 2026).

DeepSeek charges double in its peak windows, Monday to Friday 01:00-04:00 and
06:00-10:00 UTC (09:00-12:00 and 14:00-18:00 Beijing time), as third-party
copies of its pricing page reported them in October 2026; weekends are
off-peak. Whether MiMo's Token Plan or the Cline pass cap follows the same
windows is not known, so every hosted script waits for off-peak before each
log or run. Chinese public holidays are off-peak too but are not modelled
here, so the rule errs towards waiting.

    python -m opacity.offpeak            # prints peak or off-peak and the next change
    python -m opacity.offpeak wait       # returns at once if off-peak, else sleeps until it is

OFFPEAK_OVERRIDE=1 skips the wait, for an urgent run Wade has agreed to.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time

UTC = dt.timezone.utc
PEAK_WINDOWS = ((dt.time(1, 0), dt.time(4, 0)), (dt.time(6, 0), dt.time(10, 0)))  # UTC, Monday to Friday
MARGIN = dt.timedelta(minutes=15)  # a log started just before a window would run into it


def is_peak(now: dt.datetime | None = None, margin: dt.timedelta = MARGIN) -> bool:
    now = (now or dt.datetime.now(UTC)).astimezone(UTC)
    if now.weekday() >= 5:
        return False
    for start, end in PEAK_WINDOWS:
        s = dt.datetime.combine(now.date(), start, UTC) - margin
        e = dt.datetime.combine(now.date(), end, UTC)
        if s <= now < e:
            return True
    return False


def next_offpeak(now: dt.datetime | None = None) -> dt.datetime:
    t = (now or dt.datetime.now(UTC)).astimezone(UTC).replace(second=0, microsecond=0)
    while is_peak(t):
        t += dt.timedelta(minutes=1)
    return t


def wait(log=print) -> None:
    if os.environ.get("OFFPEAK_OVERRIDE") == "1":
        log("offpeak: OFFPEAK_OVERRIDE=1, not waiting")
        return
    if not is_peak():
        return
    until = next_offpeak()
    log(f"offpeak: peak window, waiting until {until:%Y-%m-%d %H:%M} UTC")
    while is_peak():
        time.sleep(60)


def main(argv: list[str]) -> int:
    if argv[:1] == ["wait"]:
        wait(lambda m: print(m, file=sys.stderr, flush=True))
        return 0
    now = dt.datetime.now(UTC)
    print(f"{now:%Y-%m-%d %H:%M} UTC: {'peak' if is_peak(now) else 'off-peak'}; "
          f"next off-peak {next_offpeak(now):%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
