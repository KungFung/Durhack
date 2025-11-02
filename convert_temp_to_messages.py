#!/usr/bin/env python3
"""
Convert a Facebook-style `temp.json` (sample provided) into a simple messages JSON
matching the format in `messages.json` (list of {sender, receiver, message, timestamp}).

Usage:
  python convert_temp_to_messages.py -i /path/to/temp.json -o /path/to/messages_converted.json
  python convert_temp_to_messages.py --overwrite  # will overwrite messages.json if you choose

The script attempts to parse any key that contains 'timestamp' as an epoch (ms or s) and
formats it as `YYYY-MM-DD HH:MM:SS` (UTC). If there are exactly two participants,
the receiver for each message will be inferred as the other participant; otherwise
receiver will be the empty string when it cannot be inferred.
"""
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List
import re


def parse_timestamp(value: Any) -> str:
    """Try to parse numeric timestamp (ms or s) into formatted UTC string.
    If parsing fails, return the original value as string.
    """
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        ts = float(value)
    else:
        s = str(value).strip()
        if s == "":
            return ""
        # remove non-digits except sign
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            ts = float(s)
        else:
            # can't parse
            return s

    # Heuristic: if timestamp looks like milliseconds (>= 10^11) assume ms
    if abs(ts) > 1e11:
        ts = ts / 1000.0
    # If it's probably seconds but too small (< 1e9), still attempt conversion
    try:
        dt = datetime.utcfromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def fix_mojibake(s: Any) -> Any:
    """Attempt to fix mojibake sequences like 'â' -> '’'.

    Strategy:
    - If ftfy is available, use it (best results).
    - Otherwise try a safe heuristic: re-encode as latin-1 then decode as utf-8.
      If that produces more non-ASCII glyphs (likely correct) we use it.
    - If the input is not a string, return as-is.
    """
    if not isinstance(s, str):
        return s

    # Fast path: nothing to fix if purely ASCII
    if all(ord(c) < 128 for c in s):
        return s

    # First apply a small hand-curated mapping for very common mojibake sequences
    # (these come from UTF-8 bytes decoded as latin-1 / CP1252 characters)
    try:
        mojimap = {
            'â': '’',
            'â': '“',
            'â': '”',
            'â': '–',
            'â': '—',
            'â€™': '’',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ãª': 'ê',
            'Ã¢': 'â',
            'Ã±': 'ñ',
            'Ã¶': 'ö',
            'Ã¼': 'ü',
            'Â ': ' ',
            '\u00e2\u0080\u0099': '’',
        }
        for k, v in mojimap.items():
            if k in s:
                s = s.replace(k, v)
    except Exception:
        pass

    # Try ftfy if available
    try:
        import ftfy

        fixed = ftfy.fix_text(s)
        if fixed:
            return fixed
    except Exception:
        # ftfy not available; fall back to heuristic
        pass

    # Heuristic re-decoding: latin-1 bytes -> utf-8 decode
    try:
        redecoded = s.encode('latin-1').decode('utf-8')
        # If redecoded contains at least as many non-ASCII characters as original,
        # assume it's better (heuristic)
        orig_nonascii = sum(1 for c in s if ord(c) > 127)
        new_nonascii = sum(1 for c in redecoded if ord(c) > 127)
        if new_nonascii >= orig_nonascii:
            return redecoded
    except Exception:
        pass

    # As a last resort try a second heuristic (sometimes strings were double-decoded)
    try:
        maybe = s.encode('utf-8', errors='replace').decode('latin-1', errors='replace')
        # choose if it produces more sensible punctuation such as ’ or —
        if '’' in maybe or '—' in maybe or '“' in maybe or '”' in maybe:
            return maybe
    except Exception:
        pass

    return s


def find_timestamp_in_obj(obj: Dict[str, Any]) -> Any:
    """Return the first value whose key contains 'timestamp' (case-insensitive)"""
    for k, v in obj.items():
        if 'timestamp' in k.lower():
            return v
    # fallback to common keys
    for key in ('time', 'ts'):
        if key in obj:
            return obj[key]
    return None


def infer_receiver(sender: str, participants: List[str]) -> str:
    sender = sender or ""
    # If exactly two participants, choose the other
    if len(participants) == 2:
        a, b = participants[0], participants[1]
        if sender == a:
            return b
        if sender == b:
            return a
        # sender not exactly matching; return other participant
        return b
    # If sender matches one of participants and more exist, return first different
    for p in participants:
        if p != sender:
            return p
    return ""


def convert(input_path: str, output_path: str, overwrite: bool = False) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}. Use --overwrite to replace.")

    # Read file in binary and try several decodings to avoid UnicodeDecodeError
    with open(input_path, 'rb') as f:
        raw = f.read()

    data = None
    for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
        try:
            text = raw.decode(enc)
            data = json.loads(text)
            break
        except Exception:
            data = None
            continue

    if data is None:
        # Last-resort: decode with replacement to avoid crashing; may mangle characters
        text = raw.decode('utf-8', errors='replace')
        data = json.loads(text)

    participants = []
    raw_parts = data.get('participants') or data.get('people') or []
    for p in raw_parts:
        if isinstance(p, dict):
            name = p.get('name') or p.get('title') or p.get('id')
            if name:
                participants.append(name)
        elif isinstance(p, str):
            participants.append(p)

    msgs = []
    for m in data.get('messages', []) or []:
        # Attempt to get sender, message content, timestamp
        sender = m.get('sender_name') or m.get('sender') or m.get('from') or ''
        content = m.get('content') or m.get('message') or m.get('text') or ''
        # Fix mojibake in textual fields
        sender = fix_mojibake(sender)
        content = fix_mojibake(content)
        raw_ts = find_timestamp_in_obj(m)
        ts = parse_timestamp(raw_ts)
        receiver = infer_receiver(sender, participants)
        msgs.append({
            'sender': sender,
            'receiver': receiver,
            'message': content,
            'timestamp': ts,
        })

    # Write JSON array
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(msgs, f, ensure_ascii=False, indent=4)


def main():
    p = argparse.ArgumentParser(description='Convert temp.json to messages-style JSON')
    p.add_argument('-i', '--input', default='temp.json', help='Path to temp.json')
    p.add_argument('-o', '--output', default='messages_converted.json', help='Output JSON path')
    p.add_argument('--overwrite', action='store_true', help='Overwrite output if present')
    args = p.parse_args()

    try:
        convert(args.input, args.output, args.overwrite)
        print(f'Wrote converted messages to: {args.output}')
    except Exception as e:
        print('Error:', e)
        raise


if __name__ == '__main__':
    main()
