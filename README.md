# SignalScout 🏈

**SignalScout** is a Streamlit-powered NFL draft prospect comparison tool. Enter a prospect's combine measurables, select their position, and get the top 10 closest historical player comps — ranked by weighted similarity score, normalized by position.

## Features

- Position-specific weighted comparison across 8 combine metrics
- Similarity % and raw distance score per comp
- Side-by-side metric display (prospect vs. database player)
- Admin panel for uploading new datasets and editing position archetypes
- Missing data handled gracefully with position-mean substitution

## Metric Encoding

Height uses NFL combine scout encoding:
- `6042` = 6 feet, 4 inches, 2/8ths = 6'4.25"
- `5102` = 5 feet, 10 inches, 2/8ths = 5'10.25"
- Last digit is **eighths of an inch**, not tenths

## Data Files

| File | Purpose |
|---|---|
| `data/players.csv` | Player database — one row per player |
| `data/archetypes.csv` | Position averages and metric priority rankings |

### players.csv columns
`name, position, height, weight, arm_length, forty, vertical, broad_jump, three_cone, shuttle`

### archetypes.csv columns
`position, height_avg, weight_avg, arm_length_avg, forty_avg, vertical_avg, broad_jump_avg, three_cone_avg, shuttle_avg, rank1, rank2, rank3, rank4`

Rank columns use exact metric names: `height, weight, arm_length, forty, vertical, broad_jump, three_cone, shuttle`

## Weight Scale

| Priority | Weight |
|---|---|
| Rank 1 | 2.0 |
| Rank 2 | 1.75 |
| Rank 3 | 1.5 |
| Rank 4 | 1.25 |
| Height & Weight (all positions) | 1.1 |
| All other metrics | 1.0 |

## Admin Setup

In Streamlit Community Cloud secrets, add:
```toml
ADMIN_PASSWORD = "Festus.4"
```

## Positions Supported
RB, WR, TE, OT, IOL, DT, EDGE, LB, CB, S, QB
