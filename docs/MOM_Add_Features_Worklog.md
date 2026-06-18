# MOM Add Features Worklog

## Branch Switch

Date: 2026-06-15

Repository:

```text
/Users/robertmauck/Documents/devel/Mauck_MOM
```

Remote:

```text
origin  https://github.com/ninthgiant/Mauck_MOM.git
```

Current branch:

```text
MOM_Add_Features
```

Tracking branch:

```text
origin/MOM_Add_Features
```

Branch switch completed by fetching from `origin` and creating a local tracking branch from `origin/MOM_Add_Features`.

Latest commit at switch:

```text
f9a6279 version 20 - change output to match Sam's process
```

## Local Repository State

After switching branches, the repository was clean except for one untracked local file:

```text
MOM_Calib_Masses.rtf
```

That file was left untouched.

## Related Arduino Data Context

The Arduino data files are CSV-style rows with this format:

```text
sensor_value, unix_time
```

The Arduino trim process preserves the full row in the trimmed file, so the visualization and analysis app should expect both the sensor value and Unix timestamp in raw and trimmed files.

## Work Log

### 2026-06-15

- Switched local repo from `MOM_Consolidated` to `MOM_Add_Features`.
- Confirmed `MOM_Add_Features` tracks `origin/MOM_Add_Features`.
- Confirmed latest branch commit is `f9a6279`.
- Preserved untracked local file `MOM_Calib_Masses.rtf`.


### 2026-06-15 UI Cursor Date/Time

- Added live cursor readout to trace plots in `MOM_Processing.py`.
- `View` plots now show row, date/time, and sensor value when the cursor moves over the trace.
- Manual marker selection plots use the same date/time readout while preserving row-number x-axis behavior for existing calculations.
- Verified `MOM_Processing.py` with `python3 -m py_compile MOM_Processing.py`.

## Next Tasks

- Inspect current Python app structure and entry point.
- Confirm how Arduino `DL*.TXT` and `TR*.TXT` files are loaded.
- Verify parser support for `sensor_value, unix_time` rows.
- Identify desired new visualization or analysis features.
- Add focused tests or sample-data checks before changing processing behavior.
