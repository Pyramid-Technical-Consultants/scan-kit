//! One typed column per requested field. Alias resolution and the G2 scale
//! happen here. The result is not a table of rows.

use std::collections::HashSet;
use std::path::Path;

use scan_kit_core::{
    column_is_integer, column_scale_factor, resolve_concept_column, resolve_requested_column,
    DERIVED_SUMS, IC3_QUAD_ZERO_FILL,
};
use serde_json::{json, Number, Value};

use super::discover::{self, Discovered};

pub fn load_columns(
    storage: &Path,
    session_id: &str,
    requested: &[String],
) -> Result<Value, String> {
    if requested.is_empty() {
        return Err("columns is required".to_owned());
    }
    if !storage.exists() {
        return Err(format!("{} does not exist", storage.display()));
    }
    let timeslices = discover::read_timeslices(storage);
    let map = discover::read_session_file(storage, session_id, "input_map.csv");
    let spots = discover::read_session_file(storage, session_id, "spot_data.csv");
    let timeslice_score = timeslices
        .first()
        .map(|bytes| score(bytes, requested))
        .unwrap_or(0);
    let map_score = map
        .as_ref()
        .map(|bytes| score(bytes, requested))
        .unwrap_or(0);
    let spot_score = spots
        .as_ref()
        .map(|bytes| score(bytes, requested))
        .unwrap_or(0);

    if timeslice_score > 0 && timeslice_score >= map_score && timeslice_score >= spot_score {
        project_files(&timeslices, requested)
    } else if map_score > 0 && map_score >= spot_score {
        project_files(std::slice::from_ref(map.as_ref().unwrap()), requested)
    } else if spot_score > 0 {
        project_files(std::slice::from_ref(spots.as_ref().unwrap()), requested)
    } else if !timeslices.is_empty() {
        project_files(&timeslices, requested)
    } else if let Some(bytes) = map.as_ref() {
        project_files(std::slice::from_ref(bytes), requested)
    } else if let Some(bytes) = spots.as_ref() {
        project_files(std::slice::from_ref(bytes), requested)
    } else {
        Err("no session csv contained the requested columns".to_owned())
    }
}

pub fn map_geom_from_spots(entry: &Discovered) -> (Option<f64>, Option<i32>) {
    let mut extent = None;
    let mut layers = None;
    for name in ["input_map.csv", "spot_data.csv"] {
        if extent.is_some() && layers.is_some() {
            break;
        }
        let Some(bytes) = discover::read_session_file(&entry.storage_path, &entry.session_id, name)
        else {
            continue;
        };
        let (file_extent, file_layers) = geom_from_csv(&bytes);
        if extent.is_none() {
            extent = file_extent;
        }
        if layers.is_none() {
            layers = file_layers;
        }
    }
    (extent, layers)
}

fn geom_from_csv(bytes: &[u8]) -> (Option<f64>, Option<i32>) {
    let Ok(headers) = header_of(bytes) else {
        return (None, None);
    };
    let energy =
        resolve_concept_column(&headers, "energy").map(|name| header_index(&headers, name));
    let x = resolve_concept_column(&headers, "x_position").map(|name| header_index(&headers, name));
    let y = resolve_concept_column(&headers, "y_position").map(|name| header_index(&headers, name));
    let mut indexes = Vec::new();
    if let Some(Some(index)) = energy {
        indexes.push(index);
    }
    if let Some(Some(index)) = x {
        indexes.push(index);
    }
    if let Some(Some(index)) = y {
        indexes.push(index);
    }
    if indexes.is_empty() {
        return (None, None);
    }
    let Ok(indexed) = read_indexed(bytes, &indexes) else {
        return (None, None);
    };
    let columns = indexed.columns;
    let mut slot = 0;
    let mut energies = None;
    if energy.and_then(|index| index).is_some() {
        energies = columns.get(slot).cloned();
        slot += 1;
    }
    let x_values = if x.and_then(|index| index).is_some() {
        let values = columns.get(slot).cloned();
        slot += 1;
        values
    } else {
        None
    };
    let y_values = if y.and_then(|index| index).is_some() {
        columns.get(slot).cloned()
    } else {
        None
    };
    let _ = slot;
    let layers = energies.as_ref().and_then(|values| {
        let mut seen = HashSet::new();
        for value in values.iter().flatten() {
            if value.is_finite() {
                let canonical = if *value == 0.0 { 0.0 } else { *value };
                seen.insert(canonical.to_bits());
            }
        }
        if seen.is_empty() {
            None
        } else {
            i32::try_from(seen.len()).ok()
        }
    });
    let mut spans = Vec::new();
    for values in [x_values, y_values].into_iter().flatten() {
        let finite: Vec<f64> = values
            .into_iter()
            .flatten()
            .filter(|value| value.is_finite())
            .collect();
        if let (Some(min), Some(max)) = (
            finite.iter().copied().reduce(f64::min),
            finite.iter().copied().reduce(f64::max),
        ) {
            let span = max - min;
            if span.is_finite() {
                spans.push(span);
            }
        }
    }
    let extent = spans.into_iter().reduce(f64::max);
    (extent, layers)
}

fn score(bytes: &[u8], requested: &[String]) -> usize {
    let Ok(headers) = header_of(bytes) else {
        return 0;
    };
    requested
        .iter()
        .filter(|name| !matches!(source_for(&headers, name), Source::Missing))
        .count()
}

fn project_files(files: &[Vec<u8>], requested: &[String]) -> Result<Value, String> {
    let Some(first) = files.first() else {
        return Err("no session csv contained the requested columns".to_owned());
    };
    let headers = header_of(first)?;
    let sources: Vec<Source> = requested
        .iter()
        .map(|name| source_for(&headers, name))
        .collect();
    if sources
        .iter()
        .all(|source| matches!(source, Source::Missing))
    {
        return Err("no session csv contained the requested columns".to_owned());
    }
    let mut series = vec![Vec::new(); requested.len()];
    for file in files {
        let file_headers = header_of(file)?;
        let file_sources: Vec<Source> = requested
            .iter()
            .map(|name| source_for(&file_headers, name))
            .collect();
        let mut indexes = Vec::new();
        for source in &file_sources {
            if let Some(list) = source.indexes() {
                for index in list {
                    if !indexes.contains(&index) {
                        indexes.push(index);
                    }
                }
            }
        }
        let (loaded, rows) = if indexes.is_empty() {
            (Vec::new(), count_rows(file)?)
        } else {
            let indexed = read_indexed(file, &indexes)?;
            let rows = indexed.columns.first().map(Vec::len).unwrap_or(0);
            (indexed.columns, rows)
        };
        let mut index_slot = std::collections::HashMap::<usize, usize>::new();
        for (slot, index) in indexes.iter().enumerate() {
            index_slot.insert(*index, slot);
        }
        for row in 0..rows {
            for (column, source) in file_sources.iter().enumerate() {
                series[column].push(value_at(source, &loaded, &index_slot, row));
            }
        }
    }
    let mut columns = Vec::new();
    let mut missing = Vec::new();
    for (index, name) in requested.iter().enumerate() {
        if matches!(sources[index], Source::Missing) {
            missing.push(name.clone());
            continue;
        }
        columns.push(column_json(name, column_is_integer(name), &series[index]));
    }
    let rows = series.iter().map(Vec::len).max().unwrap_or(0);
    Ok(json!({
        "rows": rows,
        "columns": columns,
        "missing": missing,
    }))
}

fn column_json(name: &str, integer: bool, values: &[Option<f64>]) -> Value {
    if integer {
        json!({
            "name": name,
            "dtype": "i32",
            "values": values.iter().map(|value| {
                value.and_then(|number| i32::try_from(number as i64).ok()).map(|number| json!(number)).unwrap_or(Value::Null)
            }).collect::<Vec<_>>(),
        })
    } else {
        json!({
            "name": name,
            "dtype": "f32",
            "values": values.iter().map(|value| match value {
                Some(number) if number.is_finite() => Number::from_f64((*number as f32) as f64).map(Value::Number).unwrap_or(Value::Null),
                _ => Value::Null,
            }).collect::<Vec<_>>(),
        })
    }
}

fn value_at(
    source: &Source,
    loaded: &[Vec<Option<f64>>],
    index_slot: &std::collections::HashMap<usize, usize>,
    row: usize,
) -> Option<f64> {
    match source {
        Source::Index { index, scale } => {
            let slot = index_slot.get(index)?;
            loaded
                .get(*slot)?
                .get(row)
                .copied()
                .flatten()
                .map(|value| value * scale)
        }
        Source::Derived { indexes, scale } => {
            let mut sum = 0.0;
            for index in indexes {
                let slot = index_slot.get(index)?;
                sum += loaded.get(*slot)?.get(row).copied().flatten()?;
            }
            Some(sum * scale)
        }
        Source::Zeros => Some(0.0),
        Source::Missing => None,
    }
}

enum Source {
    Index { index: usize, scale: f64 },
    Derived { indexes: Vec<usize>, scale: f64 },
    Zeros,
    Missing,
}

impl Source {
    fn indexes(&self) -> Option<Vec<usize>> {
        match self {
            Self::Index { index, .. } => Some(vec![*index]),
            Self::Derived { indexes, .. } => Some(indexes.clone()),
            Self::Zeros | Self::Missing => None,
        }
    }
}

fn source_for(headers: &[String], requested: &str) -> Source {
    if let Some(raw) = resolve_concept_column(headers, requested)
        .or_else(|| resolve_requested_column(headers, requested))
    {
        if let Some(index) = header_index(headers, raw) {
            let scale = column_scale_factor(raw).unwrap_or(1.0);
            return Source::Index { index, scale };
        }
    }
    if let Some(derived) = DERIVED_SUMS.iter().find(|item| item.target == requested) {
        let mut indexes = Vec::new();
        for source in derived.sources {
            let Some(raw) = resolve_requested_column(headers, source) else {
                indexes.clear();
                break;
            };
            let Some(index) = header_index(headers, raw) else {
                indexes.clear();
                break;
            };
            indexes.push(index);
        }
        if indexes.len() == derived.sources.len() {
            return Source::Derived {
                indexes,
                scale: derived.scale,
            };
        }
    }
    if IC3_QUAD_ZERO_FILL.contains(&requested)
        && resolve_concept_column(headers, "ic3_current_a").is_some()
    {
        return Source::Zeros;
    }
    Source::Missing
}

fn header_index(headers: &[String], raw: &str) -> Option<usize> {
    headers.iter().position(|header| header == raw)
}

fn header_of(bytes: &[u8]) -> Result<Vec<String>, String> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let headers = reader.headers().map_err(|err| err.to_string())?;
    Ok(headers.iter().map(clean_header).collect())
}

struct IndexedCsv {
    columns: Vec<Vec<Option<f64>>>,
}

fn read_indexed(bytes: &[u8], indexes: &[usize]) -> Result<IndexedCsv, String> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    // The header row is consumed so the records below are data.
    reader.headers().map_err(|err| err.to_string())?;
    let mut columns = vec![Vec::new(); indexes.len()];
    for record in reader.records() {
        let record = record.map_err(|err| err.to_string())?;
        for (slot, index) in indexes.iter().enumerate() {
            let cell = record.get(*index).and_then(parse_cell);
            columns[slot].push(cell);
        }
    }
    Ok(IndexedCsv { columns })
}

fn count_rows(bytes: &[u8]) -> Result<usize, String> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let mut rows = 0;
    for record in reader.records() {
        record.map_err(|err| err.to_string())?;
        rows += 1;
    }
    Ok(rows)
}

fn clean_header(text: &str) -> String {
    text.trim().trim_start_matches('\u{feff}').trim().to_owned()
}

fn parse_cell(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    text.parse().ok()
}
