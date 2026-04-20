# Custom `describe()` – Handling Rules

## Non-Numeric Values
- Only numeric columns are used (no name, birthday, etc.):
- No additional checks needed during calculations

## NaN (Missing Values)
- Empty values in CSV → interpreted as `NaN` (default behavior)
- `NaN` values are:
  - **ignored** in all calculations
  - **not counted** in `count`
