#!/usr/bin/env python3
"""
Split building data into Sensors, Actuators, and Configuration CSVs.

Features:
- Reads three list files of column names (one per line), supporting:
  * Exact matches
  * Wildcards '*' and '?' using fnmatch
  * Regex lines prefixed with 're:'
- Auto-detects input CSV delimiter (or use --sep to force one)
- UTF-8 safe (handles characters like 'façade')
- Loads only the required columns for memory efficiency
- Logs missing and overlapping columns; optional --allow-overlap to include columns in multiple outputs
- Optionally keeps a timestamp column in all outputs (default: 'timestamp')
- Filters the data given specific rooms (It can be either no filtering, one room or multiple ones)
- Filters the data based on a specified start and end time (eg. --start-time "3/1/2023 08:00")

Previous usage:
python split_by_rooms_category_timeframe.py --data ../../Database/AAU-BUILD-sensor.actuator/6roomsOffice/dataset_with_occupancy_delimiter_comma.csv --sensors configs/R_A_All_May/sensors.txt --actuators configs/R_A_All_May/actuators.txt --config configs/R_A_All_May/configuration.txt --outdir out/R_A_All_May --timestamp-col timestamp --rooms configs/R_A_All_May/rooms.txt --timeframe configs/R_A_All_May/timeframe.txt 

python split_by_rooms_category_timeframe.py --data ../../Database/AAU-BUILD-sensor.actuator/6roomsOffice/dataset_with_occupancy_delimiter_comma.csv --sensors configs/R_A_All_May_reduced/sensors.txt --actuators configs/R_A_All_May_reduced/actuators.txt --config configs/R_A_All_May_reduced/configuration.txt --outdir out/R_A_All_May_reduced --timestamp-col timestamp --rooms configs/R_A_All_May_reduced/rooms.txt --timeframe configs/R_A_All_May_reduced/timeframe.txt 

Sample Usage:
    python3 split_by_category.py   
        --data ../../Database/AAU-BUILD-sensor.actuator/6roomsOffice/dataset_with_occupancy_delimiter_comma.csv
        --sensors sensors.txt   
        --actuators actuators.txt   
        --config configurations.txt  
        --outdir out   
        --timestamp-col timestamp
        --timeframe timeframe.txt
        --rooms rooms.txt 

Optional flags:
    --sep ','                  # Force a delimiter (default: auto-detect)
    --allow-overlap            # Columns matching multiple categories appear in each output
    --strict                   # Exit with error if any listed columns are not found
    --case-sensitive           # Make matching case-sensitive (default is case-insensitive)
"""

import argparse
import os
import sys
import re
import fnmatch
from typing import List, Dict, Set, Tuple, Callable, Optional
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split data CSV into Sensors/Actuators/Configuration CSVs.")
    p.add_argument("--data", required=True, help="Path to the master data CSV.")
    p.add_argument("--sensors", default="sensors.txt", help="Path to sensors list file.")
    p.add_argument("--actuators", default="actuators.txt", help="Path to actuators list file.")
    p.add_argument("--config", default="configuration.txt", help="Path to configuration list file.")
    p.add_argument("--outdir", default="out", help="Output directory for the split CSV files.")
    p.add_argument("--timestamp-col", default="timestamp", help="Name of timestamp column to keep in all outputs (if present).")
    p.add_argument("--sep", default=None, help="CSV delimiter. If omitted, auto-detect.")
    p.add_argument("--allow-overlap", action="store_true", help="Allow columns to appear in multiple outputs.")
    p.add_argument("--strict", action="store_true", help="Fail if any listed columns are not found.")
    p.add_argument("--case-sensitive", action="store_true", help="Enable case-sensitive matching (default is case-insensitive).")
    p.add_argument("--rooms", default="rooms.txt", help="Path to rooms list file.")
    p.add_argument("--timeframe", default="timeframe.txt", help="Path to the file containing the desired timeframe.")
    return p.parse_args()


def load_list_file(path: str, case_sensitive: bool) -> List[str]:
    """Load list file lines, ignoring blanks and comments (#...). Returns raw patterns."""
    lines: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line if case_sensitive else line.lower())
    return lines


def compile_matchers(patterns: List[str], case_sensitive: bool) -> List[Tuple[str, Callable[[str], bool]]]:
    """
    Build matchers from patterns. Supports:
      - regex if pattern starts with 're:'
      - wildcard '*' '?' via fnmatch
      - exact match otherwise
    Returns list of (pattern, predicate).
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    matchers: List[Tuple[str, Callable[[str], bool]]] = []

    for pat in patterns:
        # Keep original for logging, but build predicate on lower if not case sensitive
        if not case_sensitive:
            src_pat = pat
        else:
            src_pat = pat

        if pat.startswith("re:"):
            regex = pat[3:]
            try:
                r = re.compile(regex, flags=flags)
            except re.error as e:
                raise ValueError(f"Invalid regex in pattern '{pat}': {e}")
            def pred_regex(c, r=r):
                return bool(r.search(c))
            matchers.append((pat, pred_regex))
        elif any(ch in pat for ch in "*?"):
            # wildcard/fnmatch
            # Compile to regex internally for performance
            # Convert to a regex pattern respecting case sensitivity
            regex = fnmatch.translate(pat)
            r = re.compile(regex, flags=flags)
            def pred_fnmatch(c, r=r):
                return bool(r.match(c))
            matchers.append((pat, pred_fnmatch))
        else:
            # exact
            def pred_exact(c, pat=pat):
                return c == pat
            matchers.append((pat, pred_exact))
    return matchers


def match_columns(all_cols: List[str], matchers: List[Tuple[str, Callable[[str], bool]]], case_sensitive: bool) -> Set[str]:
    """Return set of matching columns for the provided matchers."""
    cols = all_cols if case_sensitive else [c.lower() for c in all_cols]
    matched: Set[str] = set()
    # We must return original-cased names; make a mapping
    original_by_norm = { (c if case_sensitive else c.lower()): c for c in all_cols }

    for pat, predicate in matchers:
        for idx, cand in enumerate(cols):
            if predicate(cand):
                matched.add(original_by_norm[cand])
    return matched


def autodetect_sep(path: str) -> str:
    """Try to auto-detect separator by looking at the first non-empty line."""
    # Heuristic detection: prefers tab, semicolon, comma, pipe
    candidates = ["\t", ";", ",", "|"]
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.strip():
                sample = line
                break
        else:
            # empty file
            return ","
    counts = {sep: sample.count(sep) for sep in candidates}
    sep = max(counts, key=counts.get)
    return sep if counts[sep] > 0 else ","

# Timeframe
def load_timeframe(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Load start and end time from the timeframe.txt file.
    File format should be:
    start: 3/1/2023 13:05
    end: 3/2/2023 20:20
    """
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            # Remove comments first, then strip whitespace
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            
            if line.lower().startswith("start:"):
                start_time = line[6:].strip() # Get text after "start:"
            elif line.lower().startswith("end:"):
                end_time = line[4:].strip()   # Get text after "end:"
                
    return start_time, end_time
# -----------------------------

def load_room_list(path: str, case_sensitive: bool) -> List[str]:
    """
    Load list file for rooms. Supports multiple space-separated names per line.
    If the file is empty, it will consider all collumns.
    File format should be:
    RoomA RoomB .... (or blank)
    """
    names: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            # Remove comments first, then strip whitespace
            line_cleaned = raw.split("#", 1)[0].strip()
            if not line_cleaned:
                continue
            
            # Split on whitespace to get all names on the line
            parts = line_cleaned.split() 
            for part in parts:
                name = part.strip()
                if name:
                    names.append(name if case_sensitive else name.lower())
    return names

def sanitize_filename(text: str) -> str:
    """
    Replaces all invalid filename characters and spaces with a single underscore.
    """
    # Replace slashes, colons, and spaces
    text = re.sub(r'[\\/: ]', '_', text)
    # Remove any other characters that are not alphanumeric, underscore, or hyphen
    text = re.sub(r'[^A-Za-z0-9_-]', '', text)
    return text

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Detect or use provided sep
    sep = args.sep or autodetect_sep(args.data)
    print(f"[info] Using delimiter: {repr(sep)}")

    # Read header only to get all columns (without loading data)
    try:
        df_head = pd.read_csv(args.data, nrows=0, sep=sep, encoding="utf-8-sig", engine="python")
    except Exception as e:
        print(f"[error] Failed to read header from {args.data}: {e}", file=sys.stderr)
        sys.exit(1)

    all_cols = list(df_head.columns)
    print(f"[info] Columns detected: {len(all_cols)}")

    # Prepare case sensitivity behavior
    case_sensitive = args.case_sensitive

    # Room filtering, if any data is present in the rooms.txt
    print(f"[info] Checking for room filter file: {args.rooms}...")
    try:
        room_names = load_room_list(args.rooms, case_sensitive)
        
        if not room_names:
            print(f"[warn] Room file {args.rooms} is empty. No room filter applied.")
        else:
            print(f"[info] Filtering for columns containing: {room_names}")
            filtered_cols = set()
            
            # Always keep the timestamp column
            if args.timestamp_col in all_cols:
                filtered_cols.add(args.timestamp_col)

            # Map normalized columns to original cased columns
            original_by_norm = { (c if case_sensitive else c.lower()): c for c in all_cols }
            
            # Compile regex for selected rooms
            room_patterns = [
                re.compile(rf"^{re.escape(room)}", flags=0 if case_sensitive else re.IGNORECASE)
                for room in room_names
            ]

            for col in original_by_norm:
                col_check = col if case_sensitive else col.lower()
                is_room_col = re.match(r"^Room[A-Z]", col_check) if case_sensitive else re.match(r"^room[a-z]", col_check)
                if is_room_col:
                    # Keep only if it matches one of the selected rooms
                    if any(p.match(col_check) for p in room_patterns):
                        filtered_cols.add(col)
                else:
                    # Keep non-room-specific columns
                    filtered_cols.add(col)
            
            original_col_count = len(all_cols)
            all_cols = sorted(list(filtered_cols))  # This FILTERS all_cols
            
            print(f"[info] Room filter reduced columns from {original_col_count} to {len(all_cols)}.")
            if not all_cols:
                 print("[error] Room filter resulted in 0 columns. Aborting.", file=sys.stderr)
                 sys.exit(4)

    except FileNotFoundError:
        print(f"[info] Room file '{args.rooms}' not found.")
    except Exception as e:
        print(f"[error] Failed to process room file {args.rooms}: {e}", file=sys.stderr)
        sys.exit(1)

    # Load patterns
    sensor_patterns = load_list_file(args.sensors, case_sensitive)
    actuator_patterns = load_list_file(args.actuators, case_sensitive)
    config_patterns = load_list_file(args.config, case_sensitive)

    # Compile matchers
    sensor_matchers = compile_matchers(sensor_patterns, case_sensitive)
    actuator_matchers = compile_matchers(actuator_patterns, case_sensitive)
    config_matchers = compile_matchers(config_patterns, case_sensitive)

    # Match columns
    sensors_set = match_columns(all_cols, sensor_matchers, case_sensitive)
    actuators_set = match_columns(all_cols, actuator_matchers, case_sensitive)
    config_set = match_columns(all_cols, config_matchers, case_sensitive)

    # Report missing (based on exact tokens only, patterns are considered non-missing by design)
    # For visibility, we try to identify "exact name" lines that didn't match any header.
    def find_exact_missing(patterns: List[str], matched: Set[str]) -> List[str]:
        missing = []
        # For case-insensitive, compare normalized
        norm_headers = set(all_cols if case_sensitive else [c.lower() for c in all_cols])
        for p in patterns:
            if p.startswith("re:") or any(ch in p for ch in "*?"):
                continue
            if (p if case_sensitive else p.lower()) not in norm_headers:
                missing.append(p)
        return missing

    missing_sensors = find_exact_missing(sensor_patterns, sensors_set)
    missing_actuators = find_exact_missing(actuator_patterns, actuators_set)
    missing_config = find_exact_missing(config_patterns, config_set)

    total_missing = len(missing_sensors) + len(missing_actuators) + len(missing_config)
    if total_missing:
        print(f"[warn] Missing exact column names not found in data: {total_missing}")
        if missing_sensors:
            print(f"  - Sensors missing ({len(missing_sensors)}): {missing_sensors[:10]}{' ...' if len(missing_sensors)>10 else ''}")
        if missing_actuators:
            print(f"  - Actuators missing ({len(missing_actuators)}): {missing_actuators[:10]}{' ...' if len(missing_actuators)>10 else ''}")
        if missing_config:
            print(f"  - Configuration missing ({len(missing_config)}): {missing_config[:10]}{' ...' if len(missing_config)>10 else ''}")

    # Check overlaps
    overlaps = {
        "sensors∩actuators": sorted(sensors_set & actuators_set),
        "sensors∩config": sorted(sensors_set & config_set),
        "actuators∩config": sorted(actuators_set & config_set),
    }
    any_overlap = any(overlaps[k] for k in overlaps)
    if any_overlap:
        print("[warn] Some columns match multiple categories.")
        for k, v in overlaps.items():
            if v:
                print(f"  - {k}: {len(v)}")

    # Decide per-category columns (unique or overlapping)
    if args.allow_overlap:
        sensors_cols = sorted(sensors_set)
        actuators_cols = sorted(actuators_set)
        config_cols = sorted(config_set)
    else:
        # Enforce unique assignment with a default priority:
        # Sensors > Actuators > Configuration (you can change the order if preferred)
        assigned = set()
        sensors_cols = sorted([c for c in sensors_set if c not in assigned or (assigned.add(c) or True)])
        assigned.update(sensors_cols)
        actuators_cols = sorted([c for c in actuators_set if c not in assigned or (assigned.add(c) or True)])
        assigned.update(actuators_cols)
        config_cols = sorted([c for c in config_set if c not in assigned or (assigned.add(c) or True)])

    # Optionally keep timestamp
    ts = args.timestamp_col
    if ts in all_cols:
        if ts not in sensors_cols:
            sensors_cols = [ts] + sensors_cols
        if ts not in actuators_cols:
            actuators_cols = [ts] + actuators_cols
        if ts not in config_cols:
            config_cols = [ts] + config_cols
    else:
        print(f"[info] Timestamp column '{ts}' not found. Proceeding without it.")

    # If strict, abort on missing
    if args.strict and total_missing > 0:
        print("[error] Strict mode: aborting due to missing exact column names.", file=sys.stderr)
        sys.exit(2)

    print(f"[info] Checking for timeframe filter file: {args.timeframe}...")
    try:
        start_time, end_time = load_timeframe(args.timeframe)
        if start_time: print(f"[info] Found start time: {start_time}")
        if end_time: print(f"[info] Found end time: {end_time}")
    except FileNotFoundError:
        if args.timeframe == "timeframe.txt":
            print(f"[info] Default timeframe file 'timeframe.txt' not found. Skipping time filter.")
        else:
            print(f"[error] Timeframe file not found: {args.timeframe}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"[error] Failed to process timeframe file {args.timeframe}: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare to read only needed columns
    read_cols = set(sensors_cols) | set(actuators_cols) | set(config_cols)
    if not read_cols:
        print("[error] No columns selected to export. Check your list files and patterns.", file=sys.stderr)
        sys.exit(3)

    # Read data with selected columns only
    print(f"[info] Reading {len(read_cols)} columns from data (out of {len(all_cols)} total).")
    try:
        df = pd.read_csv(
            args.data,
            sep=sep,
            encoding="utf-8-sig",
            engine="python",
            usecols=[c for c in read_cols if c in all_cols],
            on_bad_lines="warn"
        )
    except ValueError as ve:
        # Some engines are strict with usecols; fall back to reading all and subsetting
        print(f"[warn] usecols raised {ve}; reading full file and subsetting.")
        df = pd.read_csv(args.data, sep=sep, encoding="utf-8-sig", engine="python", on_bad_lines="warn")

    if start_time or end_time:
        ts_col = args.timestamp_col
        if ts_col not in df.columns:
            print(f"Timestamp column '{ts_col}' not found in data.", file=sys.stderr)
        else:
            print(f"[info] Applying time filter on column '{ts_col}'...")
            original_rows = len(df)
            try:
                df[ts_col] = pd.to_datetime(df[ts_col])
                df = df.dropna(subset=[ts_col]) # dismiss rows that are not in the desired timeframe

                if start_time:
                    start_dt = pd.to_datetime(start_time)
                    df = df[df[ts_col] >= start_dt]
                    print(f"[info] Applied start time filter: >= {start_time}")

                if end_time:
                    end_dt = pd.to_datetime(end_time)
                    df = df[df[ts_col] <= end_dt]
                    print(f"[info] Applied end time filter: <= {end_time}") # Added this print

                filtered_rows = len(df)
                print(f"[info] Time filter applied. Rows reduced from {original_rows} to {filtered_rows}.")
                if filtered_rows == 0:
                    print("[warn] Time filter resulted in 0 rows. Output files will be empty.")
            
            except Exception as e:
                print(f"[error] Failed to apply time filter: {e}", file=sys.stderr)
                print("[info] Skipping time filtering due to error.")

    # Configure the folder name - rooms
    if not room_names:
        room_part = "all_data"
    else:
        room_part = "_".join([sanitize_filename(r) for r in room_names])

    # Time
    time_parts = []
    if start_time:
        time_parts.append(sanitize_filename(start_time))
    if end_time:
        time_parts.append(sanitize_filename(end_time))
        
    if not time_parts:
        time_part = "entire_timeframe"
    else:
        time_part = "_".join(time_parts)

    # Create the directory
    dynamic_folder_name = f"{room_part}_{time_part}"
    final_out_dir = os.path.join(args.outdir, dynamic_folder_name)
    ensure_outdir(final_out_dir)
    
    print(f"[info] Saving results to new directory: {final_out_dir}")
       
    # Subset and write outputs
    outputs = [
        ("sensors", sensors_cols),
        ("actuators", actuators_cols),
        ("configuration", config_cols),
    ]

    for name, cols in outputs:
        cols_in_df = [c for c in cols if c in df.columns]
        out_path = os.path.join(final_out_dir, f"data_{name}.csv")
        pd.DataFrame(df, columns=cols_in_df).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[ok] Wrote {name}: {out_path} (columns: {len(cols_in_df)})")

    # Write a small report
    report_path = os.path.join(final_out_dir, "split_report.txt")
    with open(report_path, "w", encoding="utf-8-sig") as rep:
        rep.write("Split Report\n")
        rep.write("====================\n")
        rep.write(f"Source file: {args.data}\n")
        rep.write(f"Delimiter: {repr(sep)}\n")
        rep.write(f"Timestamp column: {args.timestamp_col}\n")
        rep.write(f"Case sensitive: {args.case_sensitive}\n")
        rep.write(f"Allow overlap: {args.allow_overlap}\n")
        rep.write("\nCounts:\n")
        rep.write(f"  Sensors: {len(sensors_cols)} (incl. timestamp if present)\n")
        rep.write(f"  Actuators: {len(actuators_cols)} (incl. timestamp if present)\n")
        rep.write(f"  Configuration: {len(config_cols)} (incl. timestamp if present)\n")

        # Added timeframe information in the report
        start_display = start_time if start_time else "Not specified"
        end_display = end_time if end_time else "Not specified"
        rep.write(f"Start time filter: {start_display}\n")
        rep.write(f"End time filter: {end_display}\n")

        if total_missing:
            rep.write("\nMissing exact names not found in data:\n")
            if missing_sensors:
                rep.write(f"  Sensors ({len(missing_sensors)}): {', '.join(missing_sensors)}\n")
            if missing_actuators:
                rep.write(f"  Actuators ({len(missing_actuators)}): {', '.join(missing_actuators)}\n")
            if missing_config:
                rep.write(f"  Configuration ({len(missing_config)}): {', '.join(missing_config)}\n")
        if any_overlap:
            rep.write("\nOverlaps detected (before resolving):\n")
            for k, v in overlaps.items():
                if v:
                    rep.write(f"  {k} ({len(v)}): {', '.join(v)}\n")
    print(f"[ok] Wrote report: {report_path}")



if __name__ == "__main__":
    main()
