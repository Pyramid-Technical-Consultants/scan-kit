use super::ImportSpot;

pub fn parse_pld(text: &str, beam_size: f64) -> Result<Vec<ImportSpot>, String> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.is_empty() {
        return Err("PLD file is empty".into());
    }
    let header = parse_beam_header(lines[0])?;
    let mut spots = Vec::new();
    let mut plan_index = 0i32;
    let mut energy = None;
    let mut pending: Vec<(f64, f64, f64)> = Vec::new();
    let mut seen: std::collections::HashMap<(u64, u64), usize> = std::collections::HashMap::new();
    let flush = |energy: &Option<f64>,
                 pending: &mut Vec<(f64, f64, f64)>,
                 spots: &mut Vec<ImportSpot>,
                 plan_index: &mut i32| {
        let Some(energy) = *energy else {
            return;
        };
        for (x, y, weight) in pending.drain(..) {
            let charge = if weight <= 0.0 || header.cumulative <= 0.0 {
                0.0
            } else {
                weight * header.total_mu / header.cumulative
            };
            if !(charge.is_finite() && charge > 0.0) {
                continue;
            }
            spots.push(ImportSpot {
                x,
                y,
                energy,
                charge,
                beam_size,
                plan_index: *plan_index,
            });
            *plan_index += 1;
        }
    };
    for line in &lines[1..] {
        if line.starts_with("Layer,") {
            flush(&energy, &mut pending, &mut spots, &mut plan_index);
            seen.clear();
            energy = Some(parse_layer_header(line)?.0);
            continue;
        }
        if !line.starts_with("Element,") {
            return Err(format!("Unexpected PLD line: {line:?}"));
        }
        if energy.is_none() {
            return Err("PLD Element line found before any Layer header".into());
        }
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 4 {
            return Err(format!("Invalid PLD element line: {line:?}"));
        }
        let x = parse_pld_float(parts[1], "element X position")?;
        let y = parse_pld_float(parts[2], "element Y position")?;
        let weight = parse_pld_float(parts[3], "element meterset weight")?;
        // Linear find is O(n^2) over a layer of unique positions.
        let key = (x.to_bits(), y.to_bits());
        if let Some(&slot) = seen.get(&key) {
            pending[slot].2 += weight;
        } else {
            seen.insert(key, pending.len());
            pending.push((x, y, weight));
        }
    }
    flush(&energy, &mut pending, &mut spots, &mut plan_index);
    seen.clear();
    if energy.is_none() {
        return Err("PLD file contains no Layer headers".into());
    }
    if spots.is_empty() {
        return Err("No planned spots with positive MU found in PLD plan".into());
    }
    Ok(spots)
}

pub(super) struct BeamHeader {
    total_mu: f64,
    cumulative: f64,
}

pub(super) fn parse_beam_header(line: &str) -> Result<BeamHeader, String> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.first().copied() != Some("Beam") {
        return Err("PLD file must start with a Beam header line".into());
    }
    if parts.len() < 10 {
        return Err("PLD Beam header is missing required fields".into());
    }
    // The format defines total MU, cumulative weight, and layer count as the
    // last three fields — fixed indices misread any variant with extra
    // metadata columns.
    let total_mu = parse_pld_float(parts[parts.len() - 3], "beam total MU")?;
    let cumulative = parse_pld_float(parts[parts.len() - 2], "cumulative meterset weight")?;
    if !total_mu.is_finite() || !cumulative.is_finite() || total_mu <= 0.0 || cumulative <= 0.0 {
        return Err("PLD beam total MU and cumulative meterset weight must be positive".into());
    }
    Ok(BeamHeader {
        total_mu,
        cumulative,
    })
}

pub(super) fn parse_layer_header(line: &str) -> Result<(f64,), String> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.first().copied() != Some("Layer") {
        return Err("Expected a Layer header line".into());
    }
    if parts.len() < 5 {
        return Err("PLD Layer header is missing required fields".into());
    }
    let energy = parse_pld_float(parts[2], "layer energy")?;
    // Same bound the DICOM writer enforces - a hand-edited PLD is a trust
    // boundary too.
    if !energy.is_finite() || energy <= 0.0 || energy > 330.0 {
        return Err("PLD layer energy must be greater than 0 and at most 330 MeV".into());
    }
    Ok((energy,))
}

pub(super) fn parse_pld_float(value: &str, label: &str) -> Result<f64, String> {
    let number: f64 = value
        .parse()
        .map_err(|_| format!("Invalid {label}: {value:?}"))?;
    if !number.is_finite() {
        return Err(format!("Invalid {label}: {value:?}"));
    }
    Ok(number)
}
