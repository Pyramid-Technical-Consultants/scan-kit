"""Tests for configuration tuning XML document and binding logic."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scan_kit.common.app_settings import AppSettings
from scan_kit.workflows.config_tuning.labels import humanize_xml_label
from scan_kit.workflows.config_tuning.value_editors import ValueKind, format_float_display, format_value, infer_value_kind
from scan_kit.workflows.config_tuning.xml_bindings import (
    BindingSet,
    TableBinding,
    TextBinding,
    attribute_names,
    is_homogeneous_attribute_row_group,
    table_columns_for_elements,
    use_attribute_table,
)
from scan_kit.common.file_integrity import sidecar_path, verify_file_integrity, IntegrityStatus
from scan_kit.workflows.config_tuning.xml_document import XmlDocument, XmlParseError


_SCALAR_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<MapToMap type="system" version="1.0">
 <system_timeslice>0.001</system_timeslice>
 <save_datafiles>1</save_datafiles>
</MapToMap>
"""

_TABLE_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<root>
 <ion_chamber>
  <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="249.5" max_energy="250.5" K0="2.588"/>
  <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="247.0" max_energy="248.0" K0="2.633"/>
 </ion_chamber>
</root>
"""


def test_infer_value_kinds() -> None:
    assert infer_value_kind("0.001") is ValueKind.FLOAT
    assert infer_value_kind("300") is ValueKind.INT
    assert infer_value_kind("1", tag="save_datafiles") is ValueKind.BOOL
    assert infer_value_kind("0", tag="save_datafiles") is ValueKind.BOOL
    assert infer_value_kind("2.535e-08") is ValueKind.FLOAT
    assert infer_value_kind("step-and-shoot") is ValueKind.STRING


def test_format_float_display() -> None:
    assert format_float_display(2.588) == "2.588"
    assert format_float_display(300.0) == "300"
    assert format_float_display(0.001) == "0.001"
    assert format_float_display(2.535e-8) == "2.535e-08"
    assert format_float_display(393.568) == "393.568"


def test_format_value_bool_and_float() -> None:
    assert format_value(ValueKind.BOOL, True) == "1"
    assert format_value(ValueKind.FLOAT, 2.5) == "2.5"
    assert format_value(ValueKind.INT, 300) == "300"


def test_editor_minimum_width_for_scientific_notation() -> None:
    from scan_kit.workflows.config_tuning.value_editors import editor_minimum_width

    assert editor_minimum_width(ValueKind.FLOAT, "2.599e-08") >= 104
    assert editor_minimum_width(ValueKind.FLOAT, "4.0126E-7") >= 104
    assert editor_minimum_width(ValueKind.STRING, "coulombs") >= 96


def test_is_homogeneous_attribute_row_group() -> None:
    root = ET.fromstring(_TABLE_XML)
    rows = list(root.find("ion_chamber"))
    assert is_homogeneous_attribute_row_group(rows)
    assert table_columns_for_elements(rows) == [
        "in_units",
        "out_units",
        "min_energy",
        "max_energy",
        "K0",
    ]


def test_attribute_names_preserve_document_order() -> None:
    element = ET.fromstring(
        '<gain_conversion K2="-0.56" in_units="gP" K0="4.0" K1="26.81"/>'
    )
    assert attribute_names(element) == ["K2", "in_units", "K0", "K1"]


def test_table_columns_union_preserves_first_row_order_then_later_keys() -> None:
    first = ET.fromstring('<row a="1" b="2" c="3"/>')
    second = ET.fromstring('<row d="4" b="5"/>')
    assert table_columns_for_elements([first, second]) == ["a", "b", "c", "d"]


def test_bindings_round_trip_scalar(tmp_path: Path) -> None:
    path = tmp_path / "scalar.xml"
    path.write_text(_SCALAR_XML, encoding="utf-8")
    doc = XmlDocument.load(path)

    timeslice = doc.root.find("system_timeslice")
    save_flag = doc.root.find("save_datafiles")
    assert timeslice is not None and save_flag is not None

    bindings = BindingSet(
        fields=[
            TextBinding(timeslice),
            TextBinding(save_flag),
        ]
    )
    bindings.apply_field_values(["0.002", "0"])

    doc.save()
    reloaded = XmlDocument.load(path)
    assert reloaded.root.findtext("system_timeslice") == "0.002"
    assert reloaded.root.findtext("save_datafiles") == "0"


def test_bindings_round_trip_table(tmp_path: Path) -> None:
    path = tmp_path / "table.xml"
    path.write_text(_TABLE_XML, encoding="utf-8")
    doc = XmlDocument.load(path)
    chamber = doc.root.find("ion_chamber")
    assert chamber is not None
    rows = list(chamber.findall("beam_sigma_conversions"))
    table = BindingSet(
        tables=[
            TableBinding(
                elements=rows,
                columns=table_columns_for_elements(rows),
            )
        ]
    )
    updated = table.tables[0].read_row(0)
    updated["K0"] = "9.999"
    table.apply_table_values(0, [updated, table.tables[0].read_row(1)])

    doc.save()
    reloaded = XmlDocument.load(path)
    first = reloaded.root.find("ion_chamber/beam_sigma_conversions")
    assert first is not None
    assert first.get("K0") == "9.999"


def test_save_creates_backup(tmp_path: Path) -> None:
    path = tmp_path / "config.xml"
    path.write_text(_SCALAR_XML, encoding="utf-8")
    doc = XmlDocument.load(path)
    timeslice = doc.root.find("system_timeslice")
    assert timeslice is not None
    TextBinding(timeslice).write("0.005")
    doc.save()

    backup = path.with_suffix(path.suffix + ".bak")
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == _SCALAR_XML
    assert doc.root.findtext("system_timeslice") == "0.005"


def test_revert_discards_edits(tmp_path: Path) -> None:
    path = tmp_path / "config.xml"
    path.write_text(_SCALAR_XML, encoding="utf-8")
    doc = XmlDocument.load(path)
    timeslice = doc.root.find("system_timeslice")
    assert timeslice is not None
    TextBinding(timeslice).write("0.010")
    doc.mark_dirty()
    doc.revert()
    assert doc.root.findtext("system_timeslice") == "0.001"
    assert not doc.dirty


def test_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.xml"
    path.write_text("<unclosed>", encoding="utf-8")
    with pytest.raises(XmlParseError):
        XmlDocument.load(path)


def test_app_settings_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scan_kit.common.app_settings._SETTINGS_DIR",
        tmp_path,
    )
    settings = AppSettings(config_dir="/tmp/config", last_opened_xml="map2map/devices.xml")
    settings.save()
    loaded = AppSettings.load()
    assert loaded.config_dir == "/tmp/config"
    assert loaded.last_opened_xml == "map2map/devices.xml"


def test_xml_document_save_writes_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    path.write_text(_SCALAR_XML, encoding="utf-8")
    doc = XmlDocument.load(path)
    doc.mark_dirty()
    doc.save()
    result = verify_file_integrity(path)
    assert result.status == IntegrityStatus.OK
    assert sidecar_path(path).is_file()


def test_load_fixture_scan_dose_system() -> None:
    root = Path(__file__).resolve().parent.parent
    path = (
        root
        / "test_data"
        / "1943968267"
        / "1943968267"
        / "config"
        / "map2map"
        / "scan_dose_system.xml"
    )
    if not path.is_file():
        pytest.skip("fixture not available")
    doc = XmlDocument.load(path)
    assert doc.root.tag == "MapToMap"
    assert doc.root.get("type") == "system"
    geometry = doc.root.find("geometry")
    assert geometry is not None
    assert geometry.findtext("field_size_x") == "300"


def test_use_attribute_table() -> None:
    one_row = ET.fromstring(
        '<root><beam_sigma_conversions min_energy="1" max_energy="2" K0="3"/></root>'
    ).find("beam_sigma_conversions")
    two_row_xml = """\
<root>
 <beam_sigma_conversions min_energy="1" max_energy="2" K0="3"/>
 <beam_sigma_conversions min_energy="4" max_energy="5" K0="6"/>
</root>"""
    three_row_xml = """\
<root>
 <beam_sigma_conversions min_energy="1" max_energy="2" K0="3"/>
 <beam_sigma_conversions min_energy="4" max_energy="5" K0="6"/>
 <beam_sigma_conversions min_energy="7" max_energy="8" K0="9"/>
</root>"""
    assert one_row is not None
    two_rows = list(ET.fromstring(two_row_xml))
    three_rows = list(ET.fromstring(three_row_xml))
    assert not use_attribute_table([one_row])
    assert not use_attribute_table(two_rows)
    assert use_attribute_table(three_rows)


    assert humanize_xml_label("ion_chamber / gain_conversions") == "Ion Chamber / Gain Conversions"


def test_humanize_xml_label() -> None:
    assert humanize_xml_label("min_energy") == "Min Energy"
    assert humanize_xml_label("@max_energy") == "Max Energy"
    assert humanize_xml_label("beam_sigma_conversions") == "Beam Sigma Conversions"
    assert humanize_xml_label("source_to_device_distance_mm") == "Source to Device Distance mm"
    assert humanize_xml_label("system_timeslice") == "System Timeslice"
    assert humanize_xml_label("ion_chamber [2]") == "Ion Chamber [2]"
    assert humanize_xml_label("beam_sigma_conversions (89 rows)") == "Beam Sigma Conversions (89 rows)"
    assert humanize_xml_label("internal / channel (4 rows)") == "Internal / Channel (4 rows)"
    assert humanize_xml_label("K0") == "K0"
    assert humanize_xml_label("in_units") == "In Units"


_DATABASE_INTERNAL_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<MapToMap>
 <database_definitions>
  <device name="IC_1"/>
  <device name="IC_2"/>
  <device name="IC_3"/>
  <internal name="internal">
   <channel name="spot_no" type="spot_number" update=""/>
   <channel name="layer_id" type="layer_number" update=""/>
   <channel name="timestamp" units="ms" type="none"/>
   <channel name="timesliceNumber" units="none" type="timeslice"/>
  </internal>
 </database_definitions>
</MapToMap>
"""

_TRIPLE_NEST_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<MapToMap>
 <database_definitions>
  <wrapper>
   <internal name="internal">
    <channel name="a" type="t1" update=""/>
    <channel name="b" type="t2" update=""/>
    <channel name="c" type="t3" update=""/>
   </internal>
  </wrapper>
 </database_definitions>
</MapToMap>
"""


def _group_box_titles(widget) -> list[str]:
    from scan_kit.workflows.config_tuning.collapsible_group import CollapsibleGroupBox
    from PySide6.QtWidgets import QWidget

    titles: list[str] = []

    def walk(w: QWidget) -> None:
        if isinstance(w, CollapsibleGroupBox):
            titles.append(w.title())
        for child in w.children():
            if isinstance(child, QWidget):
                walk(child)

    walk(widget)
    return titles


def test_fieldset_merge_internal_channel_table() -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    from scan_kit.workflows.config_tuning.xml_form import XmlFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    root = ET.fromstring(_DATABASE_INTERNAL_XML)
    form = XmlFormWidget(root)
    titles = _group_box_titles(form)

    assert "Internal / Channel (4 rows)" in titles
    assert "Channel (4 rows)" not in titles
    assert titles.count("Internal / Channel (4 rows)") == 1
    assert app is not None


def test_fieldset_merge_triple_nested_chain() -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    from scan_kit.workflows.config_tuning.xml_form import XmlFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    root = ET.fromstring(_TRIPLE_NEST_XML)
    form = XmlFormWidget(root)
    titles = _group_box_titles(form)

    assert titles == ["Maptomap / Database Definitions / Wrapper / Internal / Channel (3 rows)"]
    assert app is not None


def test_devices_xml_uses_device_names_and_covers_leaves() -> None:
    import sys
    from pathlib import Path

    from PySide6.QtWidgets import QApplication

    from scan_kit.workflows.config_tuning.xml_document import XmlDocument
    from scan_kit.workflows.config_tuning.xml_form import XmlFormWidget

    path = (
        Path(__file__).resolve().parent.parent
        / "test_data/1943968267/1943968267/config/map2map/devices.xml"
    )
    if not path.is_file():
        pytest.skip("fixture not available")

    app = QApplication.instance() or QApplication(sys.argv)
    doc = XmlDocument.load(path)
    form = XmlFormWidget(doc.root)
    titles = _group_box_titles(form)

    assert "IC 1 X" in titles
    assert "IC 1 HCC" in titles
    assert "IC 2 X" in titles
    assert "Ion Chamber [2]" not in titles

    bound_ids: set[int] = set()
    for binding in form.bindings.fields:
        if hasattr(binding, "element"):
            bound_ids.add(id(binding.element))
    for table in form.bindings.tables:
        for element in table.elements:
            bound_ids.add(id(element))

    unbound = [
        el.tag
        for el in doc.root.iter()
        if not list(el) and id(el) not in bound_ids
    ]
    assert unbound == []
    assert app is not None


def test_all_ion_chamber_device_names_are_editable() -> None:
    import sys
    from pathlib import Path

    from PySide6.QtWidgets import QApplication

    from scan_kit.workflows.config_tuning.xml_document import XmlDocument
    from scan_kit.workflows.config_tuning.xml_form import XmlFormWidget

    path = (
        Path(__file__).resolve().parent.parent
        / "test_data/1943968267/1943968267/config/map2map/devices.xml"
    )
    if not path.is_file():
        pytest.skip("fixture not available")

    app = QApplication.instance() or QApplication(sys.argv)
    doc = XmlDocument.load(path)
    form = XmlFormWidget(doc.root)
    form.resize(800, 600)
    form.show()
    app.processEvents()

    ion_chambers = doc.root.findall(".//ion_chamber")
    for index, ic in enumerate(ion_chambers):
        device = ic.find("device")
        assert device is not None
        binding_index = next(
            idx
            for idx, binding in enumerate(form.bindings.fields)
            if getattr(binding, "element", None) is device
            and getattr(binding, "attr", None) == "name"
        )
        editor, _kind, _tag, _attr = form._field_widgets[binding_index]
        editor.setText(f"RENAMED_{index}")  # type: ignore[attr-defined]

    form.apply_to_dom()
    for index, ic in enumerate(ion_chambers):
        device = ic.find("device")
        assert device is not None
        assert device.get("name") == f"RENAMED_{index}"

    assert app is not None


def test_collapsible_fieldset_expands_and_collapses() -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QLabel

    from scan_kit.workflows.config_tuning.collapsible_group import make_collapsible_fieldset

    app = QApplication.instance() or QApplication(sys.argv)
    box, layout = make_collapsible_fieldset("Test Section")
    layout.addWidget(QLabel("inside"))
    box.show()
    app.processEvents()

    assert box.isExpanded()
    box.setExpanded(False)
    assert not box.isExpanded()
    box.setExpanded(True)
    assert box.isExpanded()
    assert app is not None


def test_attribute_fieldset_uses_inline_flow_for_many_attributes() -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QHBoxLayout

    from scan_kit.workflows.config_tuning.flow_layout import FlowWidget
    from scan_kit.workflows.config_tuning.xml_form import XmlFormWidget

    precision_xml = """\
<root>
 <precision units="coulombs" value="1.0e-12"/>
 <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="69.99" max_energy="250.01" K0="2.55665" K1="-0.0029056" K2="393.568" K3="2474.07"/>
</root>"""
    app = QApplication.instance() or QApplication(sys.argv)
    root = ET.fromstring(precision_xml)
    form = XmlFormWidget(root)
    form.resize(800, 400)
    form.show()
    app.processEvents()

    def _stacked_row_count(section_title: str) -> int:
        from scan_kit.workflows.config_tuning.collapsible_group import CollapsibleGroupBox

        for box in form.findChildren(CollapsibleGroupBox):
            if box.title() == section_title:
                layout = box.content_layout()
                stacked = 0
                for idx in range(layout.count()):
                    item = layout.itemAt(idx)
                    widget = item.widget() if item is not None else None
                    if widget is None:
                        continue
                    if isinstance(widget, FlowWidget):
                        return -1
                    if widget.layout() and isinstance(widget.layout(), QHBoxLayout):
                        stacked += 1
                return stacked
        raise AssertionError(f"missing section {section_title!r}")

    titles = _group_box_titles(form)
    assert "Precision (coulombs)" in titles
    beam_titles = [title for title in titles if title.startswith("Beam Sigma Conversions")]
    assert len(beam_titles) == 1

    assert _stacked_row_count("Precision (coulombs)") == 2
    assert _stacked_row_count(beam_titles[0]) == -1
    assert app is not None
