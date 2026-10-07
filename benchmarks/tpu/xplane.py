"""Minimal reader for ``*.xplane.pb`` profiler traces (no TensorFlow or xprof needed).

Decodes the protobuf wire format of the XSpace message (tsl/profiler/protobuf/xplane.proto):
XSpace.planes(1) -> XPlane{name(2), lines(3), event_metadata(4: map<int64, XEventMetadata>)},
XLine{name(2), events(4)}, XEvent{metadata_id(1), duration_ps(3)},
XEventMetadata{id(1), name(2), display_name(4)}. Only the fields needed to sum device time per
operation are read.
"""

from __future__ import annotations

import collections


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """Yield (field number, wire type, value) of one message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _varint(buf, i)
        elif wt == 1:
            val, i = buf[i : i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            val, i = buf[i : i + ln], i + ln
        elif wt == 5:
            val, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield num, wt, val


def device_op_times(
    path: str, plane_prefix: str = "/device:TPU:0", line_names: tuple[str, ...] = ("XLA Ops",)
) -> collections.Counter:
    """Total duration (ns) per operation name on one device plane."""
    data = open(path, "rb").read()
    totals: collections.Counter = collections.Counter()
    for num, _, plane in _fields(data):
        if num != 1:
            continue
        name, lines, meta = "", [], {}
        for f, _, v in _fields(plane):
            if f == 2:
                name = v.decode()
            elif f == 3:
                lines.append(v)
            elif f == 4:  # map entry: key=1 (int64), value=2 (XEventMetadata)
                key, md = None, None
                for ef, _, ev in _fields(v):
                    if ef == 1:
                        key = ev
                    elif ef == 2:
                        md = ev
                mname = ""
                for mf, _, mv in _fields(md or b""):
                    if mf == 2 and not mname:
                        mname = mv.decode(errors="replace")
                    elif mf == 4 and mv:
                        mname = mv.decode(errors="replace")
                meta[key] = mname
        if not name.startswith(plane_prefix):
            continue
        for line in lines:
            lname, events = "", []
            for f, _, v in _fields(line):
                if f == 2:
                    lname = v.decode()
                elif f == 4:
                    events.append(v)
            if lname not in line_names:
                continue
            for ev in events:
                mid, dur = None, 0
                for f, _, v in _fields(ev):
                    if f == 1:
                        mid = v
                    elif f == 3:
                        dur = v
                totals[meta.get(mid, str(mid))] += dur / 1000.0
    return totals
