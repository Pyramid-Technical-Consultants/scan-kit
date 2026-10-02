//! Configuration form: labels, the map2map field catalog, and the editable tree.

use serde_json::{json, Value};

use crate::xml_dom::{parse_xml, write_xml, Elem};

const INLINE_ATTRS: usize = 2;

pub fn humanize_xml_label(name: &str) -> String {
    let mut text = name.trim();
    if let Some(rest) = text.strip_prefix('@') {
        text = rest;
    }
    let mut rows_suffix = String::new();
    if let Some((head, tail)) = split_suffix(text, " rows)") {
        if let Some(open) = head.rfind(" (") {
            let count = &head[open + 2..];
            if count.chars().all(|ch| ch.is_ascii_digit()) && tail == " rows)" {
                rows_suffix = format!(" ({count} rows)");
                text = head[..open].trim();
            }
        }
    }
    let mut index_suffix = String::new();
    if let Some(open) = text.rfind(" [") {
        if text.ends_with(']') {
            let count = &text[open + 2..text.len() - 1];
            if count.chars().all(|ch| ch.is_ascii_digit()) {
                index_suffix = format!(" [{count}]");
                text = text[..open].trim();
            }
        }
    }
    if text.eq_ignore_ascii_case("value") {
        return format!("Value{index_suffix}{rows_suffix}");
    }
    if text.contains(" / ") {
        let joined = text
            .split(" / ")
            .map(|part| humanize_xml_label(part.trim()))
            .collect::<Vec<_>>()
            .join(" / ");
        return format!("{joined}{rows_suffix}{index_suffix}");
    }
    let parts: Vec<&str> = text.split('_').filter(|part| !part.is_empty()).collect();
    if parts.is_empty() {
        return name.to_owned();
    }
    let humanized = parts
        .iter()
        .enumerate()
        .map(|(index, part)| humanize_token(part, index == 0))
        .collect::<Vec<_>>()
        .join(" ");
    format!("{humanized}{index_suffix}{rows_suffix}")
}

fn split_suffix<'a>(text: &'a str, suffix: &'a str) -> Option<(&'a str, &'a str)> {
    text.strip_suffix(suffix).map(|head| (head, suffix))
}

fn humanize_token(token: &str, is_first: bool) -> String {
    if token.is_empty() {
        return String::new();
    }
    let lower = token.to_ascii_lowercase();
    if let Some(special) = special_token(&lower) {
        return special.to_owned();
    }
    if lower.starts_with('k') && lower.len() > 1 && lower[1..].chars().all(|ch| ch.is_ascii_digit())
    {
        return format!("K{}", &token[1..]);
    }
    if token.chars().all(|ch| ch.is_ascii_uppercase()) && token.chars().count() <= 4 {
        return token.to_owned();
    }
    if !is_first
        && matches!(
            lower.as_str(),
            "a" | "an" | "and" | "at" | "by" | "for" | "in" | "of" | "or" | "per" | "to" | "vs"
        )
    {
        return lower;
    }
    if token.chars().all(|ch| ch.is_ascii_digit()) {
        return token.to_owned();
    }
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut out = String::new();
    out.extend(first.to_uppercase());
    out.extend(chars.flat_map(|ch| ch.to_lowercase()));
    out
}

fn special_token(lower: &str) -> Option<&'static str> {
    Some(match lower {
        "ic" => "IC",
        "ic1" => "IC1",
        "ic2" => "IC2",
        "ic3" => "IC3",
        "hcc" => "HCC",
        "mev" => "MeV",
        "mu" => "MU",
        "gp" => "gP",
        "hv" => "HV",
        "mm" => "mm",
        "iso" => "ISO",
        "xml" => "XML",
        "dcs" => "DCS",
        "sc" => "SC",
        "rci" => "RCI",
        "k0" => "K0",
        "k1" => "K1",
        "k2" => "K2",
        "k3" => "K3",
        "k_mu" => "K MU",
        _ => return None,
    })
}

struct Field {
    name: &'static str,
    scope: &'static str,
    role: &'static str,
    note: &'static str,
    site: &'static str,
    units: Option<&'static str>,
}

const FIELDS: &[Field] = &[
    Field { name: "source_to_device_distance_mm", scope: "ion_chamber", role: "effective", note: "denominator of the strip-to-isocenter magnification factor, and the only per-chamber scale knob", site: "ion_chamber.cpp set_mag_factor", units: None },
    Field { name: "source_to_axis_distance_mm", scope: "ion_chamber", role: "overwritten", note: "seeds the magnification factor at parse time, then set_mag_factors replaces it with source_to_isocenter_distance from scan_dose_system.xml", site: "engine::load_configuration, last statement", units: None },
    Field { name: "zero_offset_at_iso_mm", scope: "ion_chamber", role: "effective", note: "added to converted positions at isocenter; not applied to spot sizes", site: "ion_chamber.hpp convert_strips", units: None },
    Field { name: "zero_offset_mm", scope: "ion_chamber", role: "unused", note: "only zero_offset_at_iso_mm is loaded", site: "ion_chamber.cpp set_data", units: None },
    Field { name: "source_to_isocenter_distance", scope: "geometry", role: "effective", note: "numerator of the magnification factor for every ion chamber, overriding each chamber's own source_to_axis_distance_mm", site: "engine.cpp load_system, devices.cpp set_mag_factors", units: None },
    Field { name: "source_to_x_axis_distance", scope: "geometry", role: "validated_only", note: "stored as m_geom.sad_x and bounds-checked, but no code reads it", site: "engine.cpp load_system", units: None },
    Field { name: "source_to_y_axis_distance", scope: "geometry", role: "validated_only", note: "stored as m_geom.sad_y and bounds-checked, but no code reads it", site: "engine.cpp load_system", units: None },
    Field { name: "magnet_axis_to_iso_distance_mm", scope: "scan_magnet", role: "magnet_effective", note: "the magnet's own SAD, used for the tangent term in dose conversions; unaffected by source_to_isocenter_distance", site: "scan_magnet.cpp set_data, engine.cpp get_sad", units: None },
    Field { name: "e0", scope: "gain_conversion", role: "unused", note: "no xmlattr getter anywhere in scan_dose", site: "scan_magnet.cpp sc_gain_conversion", units: None },
    Field { name: "e1", scope: "gain_conversion", role: "unused", note: "no xmlattr getter anywhere in scan_dose", site: "scan_magnet.cpp sc_gain_conversion", units: None },
    Field { name: "e2", scope: "gain_conversion", role: "unused", note: "no xmlattr getter anywhere in scan_dose", site: "scan_magnet.cpp sc_gain_conversion", units: None },
    Field { name: "c1", scope: "gain_conversion", role: "unused", note: "no xmlattr getter anywhere in scan_dose", site: "scan_magnet.cpp sc_gain_conversion", units: None },
    Field { name: "c3", scope: "gain_conversion", role: "unused", note: "no xmlattr getter anywhere in scan_dose", site: "scan_magnet.cpp sc_gain_conversion", units: None },
    Field { name: "d2", scope: "gain_conversion", role: "unused", note: "no xmlattr getter anywhere in scan_dose", site: "scan_magnet.cpp sc_gain_conversion", units: None },
    Field { name: "d4", scope: "gain_conversion", role: "unused", note: "no xmlattr getter anywhere in scan_dose", site: "scan_magnet.cpp sc_gain_conversion", units: None },
    Field { name: "m2", scope: "gain_conversion", role: "unused", note: "parsed but only ever written to the debug log", site: "scan_magnet.cpp sc_gain_conversion", units: None },
    Field { name: "b0", scope: "gain_conversion", role: "unused", note: "stored on volts rows, but cross-coupling reads the kilogauss row", site: "engine.cpp get_cross_correction_factor", units: Some("volts") },
    Field { name: "b1", scope: "gain_conversion", role: "unused", note: "stored on volts rows, but cross-coupling reads the kilogauss row", site: "engine.cpp get_cross_correction_factor", units: Some("volts") },
];

fn lookup<'a>(name: &str, scope: Option<&str>) -> Option<&'a Field> {
    if let Some(scope) = scope {
        if let Some(field) = FIELDS
            .iter()
            .find(|field| field.scope == scope && field.name == name)
        {
            return Some(field);
        }
    }
    let matches: Vec<&Field> = FIELDS.iter().filter(|field| field.name == name).collect();
    if matches.len() == 1 {
        Some(matches[0])
    } else {
        None
    }
}

fn tooltip(name: &str, scope: Option<&str>) -> String {
    let Some(field) = lookup(name, scope) else {
        return String::new();
    };
    let prefix = match field.role {
        "effective" => "Used by map2map",
        "magnet_effective" => "Used by map2map (scan magnet)",
        "overwritten" => "Parsed, then overwritten — no runtime effect",
        "validated_only" => "Range-checked at load, then never read",
        "unused" => "Never read by map2map",
        _ => "map2map",
    };
    format!("{prefix}: {} ({})", field.note, field.site)
}

fn is_dead(elem: &Elem, name: &str) -> bool {
    let Some(field) = FIELDS
        .iter()
        .find(|field| field.scope == elem.name && field.name == name)
    else {
        return false;
    };
    if field.role != "unused" {
        return false;
    }
    match field.units {
        Some(units) => elem
            .attr("units")
            .is_some_and(|value| value.eq_ignore_ascii_case(units)),
        None => true,
    }
}

fn child_is_dead(parent: &Elem, child_tag: &str) -> bool {
    FIELDS.iter().any(|field| {
        field.scope == parent.name && field.name == child_tag && field.role == "unused"
    })
}

pub fn config_form(xml: &str) -> Result<Value, String> {
    let root = parse_xml(xml)?;
    Ok(json!({
        "root": root.name,
        "nodes": build_element(&root, &[], &root.name, true),
    }))
}

pub fn apply_form(xml: &str, form: &Value) -> Result<String, String> {
    let mut root = parse_xml(xml)?;
    let nodes = form
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or("form nodes are required")?;
    apply_nodes(&mut root, nodes)?;
    Ok(write_xml(&root))
}

fn apply_nodes(root: &mut Elem, nodes: &[Value]) -> Result<(), String> {
    for node in nodes {
        if let Some(children) = node.get("nodes").and_then(Value::as_array) {
            apply_nodes(root, children)?;
        }
        if let Some(fields) = node.get("fields").and_then(Value::as_array) {
            for field in fields {
                let id = field
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or("field id is required")?;
                let value = field.get("value").and_then(Value::as_str).unwrap_or("");
                write_field(root, id, value)?;
            }
        }
        if node.get("kind").and_then(Value::as_str) == Some("table") {
            write_table(root, node)?;
        }
    }
    Ok(())
}

fn write_field(root: &mut Elem, id: &str, value: &str) -> Result<(), String> {
    let (path, which) = split_id(id)?;
    let elem = root
        .at_mut(&path)
        .ok_or_else(|| format!("missing XML node {id}"))?;
    if let Some(attr) = which.strip_prefix('@') {
        elem.set_attr(attr, value);
    } else {
        elem.text = value.to_owned();
    }
    Ok(())
}

fn write_table(root: &mut Elem, node: &Value) -> Result<(), String> {
    let id = node
        .get("id")
        .and_then(Value::as_str)
        .ok_or("table id is required")?;
    let (path, _) = split_id(id)?;
    let parent = root
        .at_mut(&path)
        .ok_or_else(|| format!("missing XML table {id}"))?;
    let indices: Vec<usize> = node
        .get("indices")
        .and_then(Value::as_array)
        .ok_or("table indices are required")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| "table index must be an integer".to_owned())
                .map(|index| index as usize)
        })
        .collect::<Result<_, _>>()?;
    let columns: Vec<&str> = node
        .get("columns")
        .and_then(Value::as_array)
        .ok_or("table columns are required")?
        .iter()
        .map(|value| {
            value
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "table column name is required".to_owned())
        })
        .collect::<Result<_, _>>()?;
    let rows = node
        .get("rows")
        .and_then(Value::as_array)
        .ok_or("table rows are required")?;
    if rows.len() != indices.len() {
        return Err("table row count does not match the XML".into());
    }
    for (index, row) in indices.iter().zip(rows) {
        let cells = row
            .as_array()
            .ok_or_else(|| "table row must be an array".to_owned())?;
        let child = parent
            .children
            .get_mut(*index)
            .ok_or_else(|| format!("missing table row {index}"))?;
        for (column, cell) in columns.iter().zip(cells) {
            let text = cell.as_str().unwrap_or("");
            child.set_attr(column, text);
        }
    }
    Ok(())
}

fn split_id(id: &str) -> Result<(Vec<usize>, String), String> {
    let (path, which) = if let Some((path, attr)) = id.split_once('@') {
        (path, format!("@{attr}"))
    } else if let Some((path, _)) = id.split_once("#text") {
        (path, "#text".to_owned())
    } else if let Some((path, _)) = id.split_once('#') {
        (path, String::new())
    } else {
        return Err(format!("bad field id {id}"));
    };
    if path.is_empty() {
        return Ok((Vec::new(), which));
    }
    let indexes = path
        .split('.')
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| format!("bad field id {id}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((indexes, which))
}

fn path_id(path: &[usize]) -> String {
    path.iter()
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

fn build_element(elem: &Elem, path: &[usize], title: &str, is_root: bool) -> Vec<Value> {
    if elem.children.is_empty() {
        return vec![scalar_node(elem, path, title)];
    }
    let groups = child_groups(elem);
    let sole = sole_group(elem, &groups);
    let has_own = !elem.attrs.is_empty() || (is_root && !elem.text.trim().is_empty());
    if is_root {
        if let Some((tag, indexes, kind)) = sole.as_ref() {
            let mut nodes = Vec::new();
            if has_own {
                nodes.push(attribute_fields(elem, path));
            }
            nodes.extend(place_sole(elem, path, title, tag, indexes, kind));
            return nodes;
        }
    }
    if let Some((tag, indexes, kind)) = sole.as_ref() {
        if !has_own {
            let merged = format!("{title} / {tag}");
            return place_sole(elem, path, &merged, tag, indexes, kind);
        }
        let merged = format!("{title} / {tag}");
        let mut inner = vec![attribute_fields(elem, path)];
        inner.extend(place_sole(elem, path, &merged, tag, indexes, kind));
        return vec![section(&humanize_xml_label(&merged), &inner)];
    }
    let mut inner = Vec::new();
    if has_own {
        inner.push(attribute_fields(elem, path));
    }
    inner.extend(add_groups(elem, path, &groups));
    vec![section(&humanize_xml_label(title), &inner)]
}

fn place_sole(
    elem: &Elem,
    path: &[usize],
    title: &str,
    tag: &str,
    indexes: &[usize],
    kind: &str,
) -> Vec<Value> {
    if kind == "table" {
        let label = humanize_xml_label(&format!("{title} ({} rows)", indexes.len()));
        return vec![table_node(elem, path, tag, indexes, &label)];
    }
    let index = indexes[0];
    let child = &elem.children[index];
    let child_path = child_path(path, index);
    if should_flatten(child) {
        return add_groups(child, &child_path, &child_groups(child));
    }
    build_element(child, &child_path, title, false)
}

fn add_groups(elem: &Elem, path: &[usize], groups: &[(String, Vec<usize>)]) -> Vec<Value> {
    let mut nodes = Vec::new();
    let mut index = 0;
    while index < groups.len() {
        let (tag, indexes) = &groups[index];
        if indexes.len() == 1 && is_simple_scalar(&elem.children[indexes[0]]) {
            let mut fields = Vec::new();
            while index < groups.len() {
                let (batch_tag, batch) = &groups[index];
                if batch.len() != 1 || !is_simple_scalar(&elem.children[batch[0]]) {
                    break;
                }
                fields.push(scalar_field(
                    &elem.children[batch[0]],
                    &child_path(path, batch[0]),
                    batch_tag,
                ));
                index += 1;
            }
            nodes.push(json!({ "kind": "fields", "fields": fields.concat() }));
            continue;
        }
        if use_table(elem, indexes) {
            nodes.push(table_node(
                elem,
                path,
                tag,
                indexes,
                &humanize_xml_label(&format!("{tag} ({} rows)", indexes.len())),
            ));
            index += 1;
            continue;
        }
        if is_homogeneous(elem, indexes) {
            for (offset, child_index) in indexes.iter().enumerate() {
                let child = &elem.children[*child_index];
                let row_title = inline_label(child, tag, offset);
                nodes.push(section(
                    &humanize_xml_label(&row_title),
                    &[attribute_fields(child, &child_path(path, *child_index))],
                ));
            }
            index += 1;
            continue;
        }
        if indexes.len() == 1 && elem.children[indexes[0]].children.is_empty() {
            nodes.push(scalar_node(
                &elem.children[indexes[0]],
                &child_path(path, indexes[0]),
                tag,
            ));
            index += 1;
            continue;
        }
        if indexes.len() == 1 {
            let child_index = indexes[0];
            let child = &elem.children[child_index];
            let child_path = child_path(path, child_index);
            let title = sibling_title(child, tag, 0, 1);
            if groups.len() == 1 && should_flatten(child) {
                nodes.extend(add_groups(child, &child_path, &child_groups(child)));
            } else {
                nodes.extend(build_element(child, &child_path, &title, false));
            }
            index += 1;
            continue;
        }
        for (offset, child_index) in indexes.iter().enumerate() {
            let child = &elem.children[*child_index];
            let title = sibling_title(child, tag, offset, indexes.len());
            nodes.extend(build_element(
                child,
                &child_path(path, *child_index),
                &title,
                false,
            ));
        }
        index += 1;
    }
    nodes
}

fn section(title: &str, nodes: &[Value]) -> Value {
    let collapsible = nodes.iter().any(|node| {
        matches!(
            node.get("kind").and_then(Value::as_str),
            Some("section" | "table")
        )
    });
    json!({
        "kind": "section",
        "title": title,
        "collapsible": collapsible,
        "nodes": nodes,
    })
}

fn scalar_node(elem: &Elem, path: &[usize], title: &str) -> Value {
    let fields = scalar_field(elem, path, title);
    if fields.len() > 1 || elem.attrs.len() >= INLINE_ATTRS {
        json!({
            "kind": "section",
            "title": humanize_xml_label(&inline_label(elem, &elem.name, 0)),
            "collapsible": false,
            "dead": child_dead_flag(elem),
            "nodes": [{ "kind": "fields", "fields": fields }],
        })
    } else {
        json!({ "kind": "fields", "fields": fields })
    }
}

fn scalar_field(elem: &Elem, path: &[usize], title: &str) -> Vec<Value> {
    if elem.attrs.is_empty() {
        return vec![text_field(elem, path, title)];
    }
    attribute_list(elem, path)
}

fn text_field(elem: &Elem, path: &[usize], title: &str) -> Value {
    let raw = elem.text.trim();
    json!({
        "id": format!("{}#text", path_id(path)),
        "label": humanize_xml_label(title),
        "kind": value_kind(raw),
        "value": raw,
        "dead": false,
        "tooltip": tooltip(&elem.name, None),
    })
}

fn attribute_fields(elem: &Elem, path: &[usize]) -> Value {
    let mut fields = attribute_list(elem, path);
    if !elem.text.trim().is_empty() {
        fields.push(text_field(elem, path, "value"));
    }
    json!({ "kind": "fields", "fields": fields })
}

fn attribute_list(elem: &Elem, path: &[usize]) -> Vec<Value> {
    elem.attrs
        .iter()
        .map(|(name, value)| {
            json!({
                "id": format!("{}@{name}", path_id(path)),
                "label": humanize_xml_label(name),
                "kind": value_kind(value),
                "value": value,
                "dead": is_dead(elem, name),
                "tooltip": tooltip(name, Some(&elem.name)),
            })
        })
        .collect()
}

fn table_node(elem: &Elem, path: &[usize], tag: &str, indexes: &[usize], title: &str) -> Value {
    let columns = table_columns(elem, indexes);
    let scope = elem.children[indexes[0]].name.clone();
    let column_meta: Vec<Value> = columns
        .iter()
        .map(|name| {
            json!({
                "name": name,
                "label": humanize_xml_label(name),
                "dead": is_dead(&elem.children[indexes[0]], name),
                "tooltip": tooltip(name, Some(&scope)),
            })
        })
        .collect();
    let rows: Vec<Vec<String>> = indexes
        .iter()
        .map(|index| {
            columns
                .iter()
                .map(|name| elem.children[*index].attr(name).unwrap_or("").to_owned())
                .collect()
        })
        .collect();
    json!({
        "kind": "table",
        "id": format!("{}#{tag}", path_id(path)),
        "title": title,
        "dead": indexes.iter().all(|index| child_is_dead(elem, &elem.children[*index].name)),
        "indices": indexes,
        "columns": column_meta,
        "rows": rows,
    })
}

fn table_columns(elem: &Elem, indexes: &[usize]) -> Vec<String> {
    let mut names = Vec::new();
    for index in indexes {
        for (name, _) in &elem.children[*index].attrs {
            if !names.iter().any(|have| have == name) {
                names.push(name.clone());
            }
        }
    }
    names
}

fn child_groups(elem: &Elem) -> Vec<(String, Vec<usize>)> {
    let mut groups = Vec::new();
    let mut index = 0;
    while index < elem.children.len() {
        let tag = elem.children[index].name.clone();
        let mut indexes = vec![index];
        index += 1;
        while index < elem.children.len() && elem.children[index].name == tag {
            indexes.push(index);
            index += 1;
        }
        groups.push((tag, indexes));
    }
    groups
}

fn sole_group<'a>(
    elem: &Elem,
    groups: &'a [(String, Vec<usize>)],
) -> Option<(&'a str, &'a [usize], &'static str)> {
    if groups.len() != 1 {
        return None;
    }
    let (tag, indexes) = &groups[0];
    if use_table(elem, indexes) {
        return Some((tag, indexes, "table"));
    }
    if indexes.len() == 1 && !elem.children[indexes[0]].children.is_empty() {
        return Some((tag, indexes, "container"));
    }
    None
}

fn use_table(elem: &Elem, indexes: &[usize]) -> bool {
    is_homogeneous(elem, indexes) && indexes.len() > 2
}

fn is_homogeneous(elem: &Elem, indexes: &[usize]) -> bool {
    indexes.len() >= 2
        && indexes.iter().all(|index| {
            let child = &elem.children[*index];
            child.children.is_empty() && child.text.trim().is_empty() && !child.attrs.is_empty()
        })
}

fn is_simple_scalar(elem: &Elem) -> bool {
    if !elem.children.is_empty() {
        return false;
    }
    let attrs = elem.attrs.len();
    let has_text = !elem.text.trim().is_empty();
    if attrs == 0 {
        return has_text;
    }
    !has_text && attrs <= INLINE_ATTRS
}

fn should_flatten(elem: &Elem) -> bool {
    elem.attrs.is_empty()
        && elem.text.trim().is_empty()
        && !elem.children.is_empty()
        && child_groups(elem).len() != 1
}

fn sibling_title(elem: &Elem, tag: &str, index: usize, group_len: usize) -> String {
    if let Some(name) = device_name(elem) {
        return name.to_owned();
    }
    if group_len == 1 {
        tag.to_owned()
    } else {
        format!("{tag} [{}]", index + 1)
    }
}

fn device_name(elem: &Elem) -> Option<&str> {
    let device = elem.child("device")?;
    if !device.children.is_empty() || !device.text.trim().is_empty() {
        return None;
    }
    device.attr("name").filter(|name| !name.is_empty())
}

fn inline_label(elem: &Elem, tag: &str, index: usize) -> String {
    for key in ["name", "in_units", "units", "min_energy", "out_units"] {
        if let Some(value) = elem.attr(key) {
            if !value.is_empty() {
                return format!("{tag} ({value})");
            }
        }
    }
    format!("{tag} [{}]", index + 1)
}

fn child_path(path: &[usize], index: usize) -> Vec<usize> {
    let mut next = path.to_vec();
    next.push(index);
    next
}

fn child_dead_flag(elem: &Elem) -> bool {
    child_is_dead(elem, &elem.name)
}

fn value_kind(raw: &str) -> &'static str {
    let text = raw.trim();
    if text.is_empty() {
        return "string";
    }
    if text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("false") {
        return "bool";
    }
    let lower = text.to_ascii_lowercase();
    if !lower.contains('.') && !lower.contains('e') && text.parse::<i64>().is_ok() {
        return "int";
    }
    if text.parse::<f64>().is_ok() {
        "float"
    } else {
        "string"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_match_the_python_catalog() {
        assert_eq!(humanize_xml_label("min_energy"), "Min Energy");
        assert_eq!(humanize_xml_label("@max_energy"), "Max Energy");
        assert_eq!(
            humanize_xml_label("beam_sigma_conversions"),
            "Beam Sigma Conversions"
        );
        assert_eq!(
            humanize_xml_label("source_to_device_distance_mm"),
            "Source to Device Distance mm"
        );
        assert_eq!(humanize_xml_label("ion_chamber [2]"), "Ion Chamber [2]");
        assert_eq!(
            humanize_xml_label("beam_sigma_conversions (89 rows)"),
            "Beam Sigma Conversions (89 rows)"
        );
        assert_eq!(
            humanize_xml_label("internal / channel (4 rows)"),
            "Internal / Channel (4 rows)"
        );
        assert_eq!(humanize_xml_label("K0"), "K0");
        assert_eq!(
            humanize_xml_label("ion_chamber / gain_conversions"),
            "Ion Chamber / Gain Conversions"
        );
    }

    #[test]
    fn three_attribute_rows_become_a_table_and_a_device_name_is_the_title() {
        let xml = r#"
            <devices>
              <ion_chamber>
                <device name="IC_1_X"/>
                <zero_offset_mm>1</zero_offset_mm>
                <beam_sigma_conversions min_energy="70" K0="1"/>
                <beam_sigma_conversions min_energy="80" K0="2"/>
                <beam_sigma_conversions min_energy="90" K0="3"/>
              </ion_chamber>
            </devices>
        "#;
        let form = config_form(xml).unwrap();
        let text = form.to_string();
        assert!(text.contains("IC_1_X"));
        assert!(text.contains("\"kind\":\"table\""));
        let edited = apply_form(
            xml,
            &json!({
                "nodes": [{
                    "kind": "table",
                    "id": find_table_id(&form),
                    "indices": find_indices(&form),
                    "columns": [{"name": "K0"}],
                    "rows": [["9"], ["8"], ["7"]]
                }]
            }),
        )
        .unwrap();
        assert!(edited.contains("K0=\"9\""));
        assert!(edited.contains("K0=\"7\""));
    }

    fn find_table_id(form: &Value) -> String {
        fn walk(node: &Value) -> Option<String> {
            if node.get("kind").and_then(Value::as_str) == Some("table") {
                return node.get("id").and_then(Value::as_str).map(str::to_owned);
            }
            for child in node
                .get("nodes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(found) = walk(child) {
                    return Some(found);
                }
            }
            None
        }
        walk(form).unwrap()
    }

    fn find_indices(form: &Value) -> Value {
        fn walk(node: &Value) -> Option<Value> {
            if node.get("kind").and_then(Value::as_str) == Some("table") {
                return node.get("indices").cloned();
            }
            for child in node
                .get("nodes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(found) = walk(child) {
                    return Some(found);
                }
            }
            None
        }
        walk(form).unwrap()
    }
}
