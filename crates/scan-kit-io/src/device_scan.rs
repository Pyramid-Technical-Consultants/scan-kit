//! Discover IGX devices on a local IPv4 subnet.
//!
//! `scan_subnet` expands a CIDR into host addresses and probes each with
//! `GET /io/admin/version/value.json`, the same probe `runner::connect` uses.
//! The probe is injectable so the scan loop is unit-testable without hardware.

use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

const MAX_HOSTS: u64 = 1024;
const SCAN_WORKERS: usize = 64;
const PROBE_TIMEOUT: Duration = Duration::from_millis(1200);

/// One reachable IGX device found on the network.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    pub host: String,
    pub version: Value,
    pub device_type: Option<Value>,
}

/// Scan a CIDR subnet for IGX devices and return the tool result payload.
pub fn scan_subnet(cidr: &str) -> Result<Value, String> {
    let hosts = subnet_hosts(cidr)?;
    let found = scan(&hosts, &probe_device);
    Ok(json!({
        "devices": found
            .iter()
            .map(|info| json!({
                "host": info.host,
                "version": info.version,
                "device_type": info.device_type.clone().unwrap_or(Value::Null),
            }))
            .collect::<Vec<_>>()
    }))
}

/// Probe one host for the IGX version and device type resources.
fn probe_device(host: &str) -> Option<DeviceInfo> {
    let version = crate::runner::http_json_timed(host, "admin/version", PROBE_TIMEOUT)?;
    let device_type = crate::runner::http_json_timed(host, "admin/device_type", PROBE_TIMEOUT);
    Some(DeviceInfo {
        host: host.to_owned(),
        version,
        device_type,
    })
}

/// Expand `192.168.100.0/24` into host addresses, or a bare IP into itself.
fn subnet_hosts(cidr: &str) -> Result<Vec<String>, String> {
    let text = cidr.trim();
    let (addr, prefix) = match text.split_once('/') {
        Some((addr, prefix)) => (addr.trim(), prefix.trim()),
        None => (text, "32"),
    };
    let base: Ipv4Addr = addr
        .parse()
        .map_err(|_| format!("invalid subnet: {cidr}"))?;
    let prefix: u32 = prefix
        .parse()
        .map_err(|_| format!("invalid subnet: {cidr}"))?;
    if prefix > 32 {
        return Err(format!("invalid subnet: {cidr}"));
    }
    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - prefix)
    };
    let network = u32::from(base) & mask;
    let broadcast = network | !mask;
    let (first, last) = match prefix {
        // RFC 3021 point-to-point links and single hosts have no reserved ends.
        31 | 32 => (network, broadcast),
        _ => (network + 1, broadcast.saturating_sub(1)),
    };
    let count = u64::from(last) - u64::from(first) + 1;
    if count > MAX_HOSTS {
        return Err(format!(
            "subnet too large ({count} hosts); use a /22 or smaller"
        ));
    }
    Ok((first..=last)
        .map(|ip| Ipv4Addr::from(ip).to_string())
        .collect())
}

/// Probe every host across a shared cursor of scoped worker threads.
fn scan(hosts: &[String], probe: &(dyn Fn(&str) -> Option<DeviceInfo> + Sync)) -> Vec<DeviceInfo> {
    let cursor = AtomicUsize::new(0);
    let found = Mutex::new(Vec::new());
    thread::scope(|scope| {
        for _ in 0..SCAN_WORKERS.min(hosts.len()) {
            scope.spawn(|| loop {
                let index = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(host) = hosts.get(index) else { break };
                if let Some(info) = probe(host) {
                    if let Ok(mut hits) = found.lock() {
                        hits.push(info);
                    }
                }
            });
        }
    });
    let mut found = found.into_inner().unwrap_or_default();
    found.sort_by_key(|info| host_key(&info.host));
    found
}

/// Numeric IPv4 key so scanned results list in network order.
fn host_key(host: &str) -> u32 {
    host.split(':')
        .next()
        .and_then(|text| text.parse::<Ipv4Addr>().ok())
        .map(u32::from)
        .unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use serde_json::json;

    use super::{probe_device, scan, scan_subnet, subnet_hosts, DeviceInfo};

    #[test]
    fn a_cidr_expands_to_its_hosts() {
        let hosts = subnet_hosts("192.168.100.0/24").unwrap();
        assert_eq!(hosts.len(), 254);
        assert_eq!(hosts.first().unwrap(), "192.168.100.1");
        assert_eq!(hosts.last().unwrap(), "192.168.100.254");
        assert_eq!(subnet_hosts("10.0.0.0/30").unwrap().len(), 2);
        assert_eq!(subnet_hosts("10.0.0.4/31").unwrap().len(), 2);
        assert_eq!(subnet_hosts("10.0.0.9/32").unwrap(), ["10.0.0.9"]);
        assert_eq!(subnet_hosts("10.0.0.9").unwrap(), ["10.0.0.9"]);
        // strict=False semantics: host bits are masked off.
        assert_eq!(subnet_hosts("10.0.0.9/30").unwrap().len(), 2);
    }

    #[test]
    fn a_subnet_over_1024_hosts_is_rejected() {
        assert!(subnet_hosts("10.0.0.0/22").unwrap().len() <= 1024);
        assert!(subnet_hosts("10.0.0.0/21")
            .unwrap_err()
            .contains("too large"));
        assert!(subnet_hosts("nope/24").is_err());
        assert!(subnet_hosts("10.0.0.0/33").is_err());
    }

    #[test]
    fn scan_keeps_only_devices_in_numeric_order() {
        let hosts = vec![
            "10.0.0.10".to_string(),
            "10.0.0.2".to_string(),
            "10.0.0.1".to_string(),
            "10.0.0.3".to_string(),
        ];
        let probed = AtomicUsize::new(0);
        let found = scan(&hosts, &|host| {
            probed.fetch_add(1, Ordering::SeqCst);
            if host == "10.0.0.1" || host == "10.0.0.10" {
                Some(DeviceInfo {
                    host: host.to_owned(),
                    version: json!("2.1"),
                    device_type: None,
                })
            } else {
                None
            }
        });
        assert_eq!(probed.load(Ordering::SeqCst), 4);
        assert_eq!(
            found
                .iter()
                .map(|info| info.host.as_str())
                .collect::<Vec<_>>(),
            ["10.0.0.1", "10.0.0.10"]
        );
    }

    #[test]
    fn probe_device_reads_version_and_device_type() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let bodies = [json!("3.2.1").to_string(), json!("ix256").to_string()];
            for body in bodies {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 2048];
                let _ = stream.read(&mut request);
                let reply = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(reply.as_bytes()).unwrap();
            }
        });
        let info = probe_device(&format!("127.0.0.1:{port}")).unwrap();
        assert_eq!(info.version, json!("3.2.1"));
        assert_eq!(info.device_type, Some(json!("ix256")));
    }

    #[test]
    fn probe_device_returns_none_when_the_host_is_dead() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        assert!(probe_device(&format!("127.0.0.1:{port}")).is_none());
    }

    #[test]
    fn scan_subnet_reports_a_devices_list() {
        let result = scan_subnet("127.0.0.1").unwrap();
        assert_eq!(result["devices"], json!([]));
    }
}
