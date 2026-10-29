# Data quality report

## Room A

| quantity | count |
|---|---:|
| raw_records | 234019 |
| rejected_not_csi | 0 |
| rejected_bad_length | 0 |
| rejected_zero | 0 |
| rejected_duplicate | 0 |
| observations | 234019 |

Band coverage of kept observations:

- full: 67572 (28.9%)
- low_20mhz: 165200 (70.6%)
- partial: 1247 (0.5%)
- empty: 0 (0.0%)

Truncated captures (complete packets before the truncation were kept):

- `Experiment-2/realdata/input_data/30ms/csidata/sit/sit109.pcap`: 178 observations kept
- `Experiment-2/realdata/input_data/30ms/csidata/sit/sit144.pcap`: 178 observations kept

Captures with fewer than 150 valid observations (no window): 12

- `A__development__sit__sit117` (development): 115
- `A__development__sit__sit118` (development): 128
- `A__development__sit__sit126` (development): 115
- `A__development__sit__sit134` (development): 131
- `A__development__sit__sit138` (development): 122
- `A__development__sit__sit176` (development): 145
- `A__development__sit__sit5` (development): 112
- `A__development__sitdown__sitdown136` (development): 115
- `A__development__sitdown__sitdown155` (development): 148
- `A__validation__sit__sit155` (validation): 128
- `A__validation__sit__sit201` (validation): 112
- `A__validation__walk__walk37` (validation): 121

Windows per label and split:

| label | development | validation |
|---|---|---|
| FALL | 177 | 25 |
| SIT | 171 | 23 |
| SITDOWN | 153 | 25 |
| STAND | 115 | 25 |
| STANDUP | 155 | 25 |
| WALK | 154 | 24 |

## Room B

| quantity | count |
|---|---:|
| raw_records | 600500 |
| rejected_not_csi | 0 |
| rejected_bad_length | 0 |
| rejected_zero | 0 |
| rejected_duplicate | 0 |
| observations | 600500 |

Band coverage of kept observations:

- full: 562010 (93.6%)
- low_20mhz: 586 (0.1%)
- partial: 37904 (6.3%)
- empty: 0 (0.0%)

Captures with fewer than 150 valid observations (no window): 0


Windows per label and split:

| label | development | validation |
|---|---|---|
| EMPTY | 726 | 75 |
| LYING | 978 | 75 |
| SIT | 384 | 75 |
| STAND | 588 | 75 |
| WALK | 552 | 75 |
