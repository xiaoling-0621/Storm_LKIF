# Data Summary

- Matched storms: 1654
- ERA5 sample file: `data\era5\1972\Rita\TCND_Rita_1972070500_sst_z_u_v.nc`
- Sample storm_id format: `1972:RITA`
- Intensity timestamp column: `timestamp`
- Target variable: `wind` -> `wind_like_feature`
- Target mode: `delta`
- EOF fit scope: `global_selected`
- Causality scope: `global_segmented`
- Inference note: Columns 4/5 are inferred normalized pressure/wind features; column 5 is used as wind-like intensity and column 4 as pressure-like intensity.

## ERA5 Structure

- Dims: `{'time': 1, 'pressure_level': 4, 'latitude': 81, 'longitude': 81}`
- Variables: `['u', 'v', 'z', 'sst']`
- Expanded channels: `['u_200', 'u_500', 'u_850', 'u_925', 'v_200', 'v_500', 'v_850', 'v_925', 'z_200', 'z_500', 'z_850', 'z_925', 'sst']`

## Delta Target Alignment

- `delta` mode uses `target[t+1] - target[t]`.
- ERA5 features are truncated to the first `T-1` timestamps.
- The mapping is `ERA5(t) -> target(t+1) - target(t)`.

## Storm Counts

| storm_id | era_time_count | intensity_time_count | intensity_time_min | intensity_time_max |
| --- | ---: | ---: | --- | --- |
| 1972:RITA | 100 | 100 | 1972-07-05 00:00:00 | 1972-07-30 00:00:00 |
| 1963:BESS | 87 | 87 | 1963-07-25 18:00:00 | 1963-08-16 06:00:00 |
| 1986:WAYNE | 87 | 87 | 1986-08-16 00:00:00 | 1986-09-06 12:00:00 |
| 1967:OPAL | 85 | 85 | 1967-08-29 00:00:00 | 1967-09-19 00:00:00 |
| 1965:JEAN | 84 | 84 | 1965-07-24 00:00:00 | 1965-08-13 18:00:00 |
| 1983:ABBY | 81 | 81 | 1983-08-04 00:00:00 | 1983-08-24 00:00:00 |
| 1964:KATHY | 76 | 76 | 1964-08-10 18:00:00 | 1964-08-29 12:00:00 |
| 1991:NAT | 76 | 76 | 1991-09-14 00:00:00 | 1991-10-02 18:00:00 |
| 1968:MARY | 75 | 75 | 1968-07-19 00:00:00 | 1968-08-06 12:00:00 |
| 1953:NINA | 73 | 73 | 1953-08-07 06:00:00 | 1953-08-27 00:00:00 |
| 1986:VERA | 72 | 72 | 1986-08-13 12:00:00 | 1986-08-31 06:00:00 |
| 1971:OLIVE | 70 | 70 | 1971-07-24 06:00:00 | 1971-08-10 12:00:00 |
| 2009:PARMA | 69 | 69 | 2009-09-27 18:00:00 | 2009-10-14 18:00:00 |
| 1994:VERNE | 68 | 68 | 1994-10-16 00:00:00 | 1994-11-01 18:00:00 |
| 2014:HALONG | 68 | 68 | 2014-07-28 00:00:00 | 2014-08-13 18:00:00 |
| 1976:IRIS | 67 | 67 | 1976-09-12 18:00:00 | 1976-10-01 06:00:00 |
| 1992:OMAR | 67 | 67 | 1992-08-23 00:00:00 | 1992-09-08 18:00:00 |
| 1951:IRIS | 66 | 66 | 1951-04-28 12:00:00 | 1951-05-14 18:00:00 |
| 1955:HOPE | 66 | 66 | 1955-08-02 18:00:00 | 1955-08-19 00:00:00 |
| 1982:CECIL | 66 | 66 | 1982-08-02 00:00:00 | 1982-08-19 00:00:00 |
