# Dataset card

## Default: virtual facility
- Created by `src/data.py`, seed 42, default 365 days starting 2025-01-01.
- Unit: hourly consumed energy in kWh, not instantaneous kW.
- Simulates base load, operating-hour load, weekday/weekend behavior, seasonal variation and random noise.
- Injects 20 sensor gaps and one duplicate to exercise data quality checks.
- This is entirely synthetic. Its accuracy is not evidence of industrial performance.
- No genuine weather, employee schedules, outages, or equipment meters are included.
- Training and inference deliberately do not receive the simulator's equations or noise values.

## Optional public measured dataset
UCI Individual Household Electric Power Consumption:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Download:
https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip

Citation: Hebrail, G., & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

License: Creative Commons Attribution 4.0 (CC BY 4.0). Attribute the source and identify hourly conversion and missing-data handling as modifications.

The source describes 2,075,259 minute observations from one home in Sceaux, France, December 2006-November 2010. It is not a manufacturing/data-center dataset.

`Global_active_power` is minute-averaged kW. One minute's energy = power / 60 kWh. The loader deduplicates minute timestamps by mean, then sums minute energy for hours with all 60 valid readings. Partial hours become missing. It does not incorrectly sum kW values or claim partial-hour energy is complete.

Extract the TXT file into `data/raw/`, then run:
```
python main.py --mode train --uci-file data/raw/household_power_consumption.txt
```
The original file ends on a partial day; `--mode train` works without requiring a complete final day. For a future forecast, trim a COPY of the original minute source to end at the last complete 23:59 minute with adequate measured history, and run `--mode all` on that trimmed source. Keep the unmodified download separately and document the trimming. Do not reload a cleaned export as raw data without restoring imputed hours to missing: doing so would mistakenly treat invented labels as observed.

A separate measured-data experiment was executed and is stored in `outputs/public_uci/`. Run `python public_experiment.py --file data/raw/household_power_consumption.txt` to reproduce it without replacing the synthetic outputs. The original raw data download is not included in the package. Do not mix public-data and synthetic result claims.

## Custom CSV contract
```
timestamp,energy_kwh
2025-01-01 00:00:00,42.7
2025-01-01 01:00:00,40.1
```
One facility, one series, one timezone-naive hourly timeline. Timestamp denotes hour start. Value describes energy in that complete hour; the forecast origin is operationally available only after the hour ends. At least 60 eligible forecast days are required; 180-365 days or more are preferable. Hour-start stamps can be UTC represented consistently as naive UTC. Local daylight-saving transitions require explicit preprocessing; do not silently merge different repeated local hours.
