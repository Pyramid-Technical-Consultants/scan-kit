//! A small element tree for configuration XML.
//!
//! Comments and the XML declaration are dropped on parse. The writer emits a
//! declaration and preserves attribute order.

#[derive(Clone, Debug, PartialEq)]
pub struct Elem {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub text: String,
    pub children: Vec<Elem>,
}

impl Elem {
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    pub fn set_attr(&mut self, name: &str, value: &str) {
        if let Some((_, slot)) = self.attrs.iter_mut().find(|(key, _)| key == name) {
            *slot = value.to_owned();
        } else {
            self.attrs.push((name.to_owned(), value.to_owned()));
        }
    }

    pub fn child(&self, name: &str) -> Option<&Elem> {
        self.children.iter().find(|child| child.name == name)
    }

    pub fn at_mut(&mut self, path: &[usize]) -> Option<&mut Elem> {
        let mut node = self;
        for index in path {
            node = node.children.get_mut(*index)?;
        }
        Some(node)
    }
}

pub fn parse_xml(input: &str) -> Result<Elem, String> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        index: 0,
    };
    parser.skip_misc()?;
    if parser.done() {
        return Err("XML document is empty.".into());
    }
    let root = parser.element()?;
    parser.skip_misc()?;
    if !parser.done() {
        return Err("XML document has trailing content.".into());
    }
    Ok(root)
}

pub fn write_xml(root: &Elem) -> String {
    let mut out = String::from("<?xml version='1.0' encoding='utf-8'?>\n");
    write_elem(&mut out, root);
    out.push('\n');
    out
}

fn write_elem(out: &mut String, elem: &Elem) {
    out.push('<');
    out.push_str(&elem.name);
    for (key, value) in &elem.attrs {
        out.push(' ');
        out.push_str(key);
        out.push_str("=\"");
        out.push_str(&escape_attr(value));
        out.push('"');
    }
    let text = elem.text.trim();
    if elem.children.is_empty() && text.is_empty() {
        out.push_str("/>");
        return;
    }
    out.push('>');
    if !text.is_empty() {
        out.push_str(&escape_text(text));
    }
    for child in &elem.children {
        write_elem(out, child);
    }
    out.push_str("</");
    out.push_str(&elem.name);
    out.push('>');
}

fn escape_text(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    out
}

fn escape_attr(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

struct Parser<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl Parser<'_> {
    fn done(&self) -> bool {
        self.index >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.index).copied()
    }

    fn bump(&mut self) -> Result<u8, String> {
        let byte = self
            .peek()
            .ok_or_else(|| "unexpected end of XML".to_owned())?;
        self.index += 1;
        Ok(byte)
    }

    fn skip_misc(&mut self) -> Result<(), String> {
        loop {
            self.skip_ws();
            if self.starts_with(b"<?") {
                self.skip_until(b"?>")?;
                continue;
            }
            if self.starts_with(b"<!--") {
                self.skip_until(b"-->")?;
                continue;
            }
            if self.starts_with(b"<!DOCTYPE") || self.starts_with(b"<![CDATA[") {
                return Err("unsupported XML declaration".into());
            }
            break;
        }
        Ok(())
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.index += 1;
        }
    }

    fn starts_with(&self, needle: &[u8]) -> bool {
        self.bytes[self.index..].starts_with(needle)
    }

    fn skip_until(&mut self, needle: &[u8]) -> Result<(), String> {
        if let Some(found) = self.bytes[self.index..]
            .windows(needle.len())
            .position(|window| window == needle)
        {
            self.index += found + needle.len();
            Ok(())
        } else {
            Err("unterminated XML markup".into())
        }
    }

    fn element(&mut self) -> Result<Elem, String> {
        self.expect(b'<')?;
        if matches!(self.peek(), Some(b'/' | b'!' | b'?')) {
            return Err("expected an XML element".into());
        }
        let name = self.name()?;
        let mut attrs = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'/') => {
                    self.bump()?;
                    self.expect(b'>')?;
                    return Ok(Elem {
                        name,
                        attrs,
                        text: String::new(),
                        children: Vec::new(),
                    });
                }
                Some(b'>') => {
                    self.bump()?;
                    break;
                }
                Some(_) => {
                    let key = self.name()?;
                    self.skip_ws();
                    self.expect(b'=')?;
                    self.skip_ws();
                    let value = self.quoted()?;
                    attrs.push((key, value));
                }
                None => return Err("unterminated XML start tag".into()),
            }
        }
        let mut text = String::new();
        let mut children = Vec::new();
        loop {
            if self.peek().is_none() {
                return Err(format!("unterminated XML element <{name}>"));
            }
            if self.starts_with(b"</") {
                self.index += 2;
                let end = self.name()?;
                self.skip_ws();
                self.expect(b'>')?;
                if end != name {
                    return Err(format!("expected </{name}>, found </{end}>"));
                }
                break;
            }
            if self.starts_with(b"<!--") {
                self.skip_until(b"-->")?;
                continue;
            }
            if self.starts_with(b"<?") {
                self.skip_until(b"?>")?;
                continue;
            }
            if self.starts_with(b"<![CDATA[") {
                self.index += b"<![CDATA[".len();
                let start = self.index;
                self.skip_until(b"]]>")?;
                let end = self.index - b"]]>".len();
                text.push_str(
                    std::str::from_utf8(&self.bytes[start..end]).map_err(|err| err.to_string())?,
                );
                continue;
            }
            if self.peek() == Some(b'<') {
                children.push(self.element()?);
                continue;
            }
            text.push_str(&self.text_run()?);
        }
        Ok(Elem {
            name,
            attrs,
            text,
            children,
        })
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        let found = self.bump()?;
        if found != byte {
            return Err(format!(
                "expected '{}', found '{}'",
                byte as char, found as char
            ));
        }
        Ok(())
    }

    fn name(&mut self) -> Result<String, String> {
        let start = self.index;
        while matches!(
            self.peek(),
            Some(b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b':' | b'-' | b'.')
        ) {
            self.index += 1;
        }
        if start == self.index {
            return Err("expected an XML name".into());
        }
        std::str::from_utf8(&self.bytes[start..self.index])
            .map(|name| name.to_owned())
            .map_err(|err| err.to_string())
    }

    fn quoted(&mut self) -> Result<String, String> {
        let quote = self.bump()?;
        if quote != b'"' && quote != b'\'' {
            return Err("expected a quoted XML attribute".into());
        }
        let mut out = String::new();
        loop {
            let byte = self.bump()?;
            if byte == quote {
                return Ok(out);
            }
            if byte == b'&' {
                out.push(self.entity()?);
                continue;
            }
            out.push(byte as char);
        }
    }

    fn text_run(&mut self) -> Result<String, String> {
        let mut out = String::new();
        while let Some(byte) = self.peek() {
            if byte == b'<' {
                break;
            }
            self.bump()?;
            if byte == b'&' {
                out.push(self.entity()?);
            } else {
                out.push(byte as char);
            }
        }
        Ok(out)
    }

    fn entity(&mut self) -> Result<char, String> {
        let start = self.index;
        while self.peek().is_some_and(|byte| byte != b';') {
            self.index += 1;
            if self.index - start > 16 {
                return Err("unterminated XML entity".into());
            }
        }
        self.expect(b';')?;
        let name = std::str::from_utf8(&self.bytes[start..self.index - 1])
            .map_err(|err| err.to_string())?;
        match name {
            "amp" => Ok('&'),
            "lt" => Ok('<'),
            "gt" => Ok('>'),
            "quot" => Ok('"'),
            "apos" => Ok('\''),
            _ if name.starts_with('#') => decode_numeric(name),
            _ => Err(format!("unknown XML entity &{name};")),
        }
    }
}

fn decode_numeric(name: &str) -> Result<char, String> {
    let digits = name.as_bytes();
    let value = if digits.get(1) == Some(&b'x') || digits.get(1) == Some(&b'X') {
        u32::from_str_radix(name.get(2..).unwrap_or(""), 16)
    } else {
        name.get(1..).unwrap_or("").parse::<u32>()
    };
    value
        .ok()
        .and_then(char::from_u32)
        .ok_or_else(|| format!("bad XML character reference &{name};"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_keep_document_order_and_round_trip() {
        let xml = r#"<?xml version="1.0"?><root a="1" b="&amp;"><child c='x'/></root>"#;
        let root = parse_xml(xml).unwrap();
        assert_eq!(
            root.attrs,
            vec![("a".into(), "1".into()), ("b".into(), "&".into())]
        );
        assert_eq!(root.children[0].attr("c"), Some("x"));
        let again = parse_xml(&write_xml(&root)).unwrap();
        assert_eq!(again.attr("b"), Some("&"));
        assert_eq!(again.children[0].attr("c"), Some("x"));
    }

    #[test]
    fn a_broken_tag_is_rejected() {
        assert!(parse_xml("<unclosed>").is_err());
    }
}
