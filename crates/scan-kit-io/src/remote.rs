//! Remote readers. A fixture stands in for a server in tests.
//!
//! SFTP is synchronous `ssh2`. FTP and FTPS use `suppaftp`. An http URL is one
//! archive. `smb://` is turned into a UNC path before this module sees it.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use crate::location::{self, Remote};

pub(crate) struct Item {
    pub name: String,
    pub dir: bool,
}

pub(crate) struct Stat {
    pub dir: bool,
    pub size: u64,
    pub mtime_ns: i64,
}

pub(crate) trait Fs {
    fn list(&mut self, path: &str) -> Result<Vec<Item>, String>;
    fn read(&mut self, path: &str) -> Result<Option<Vec<u8>>, String>;
    fn stat(&mut self, path: &str) -> Result<Option<Stat>, String>;
    fn copy_to(&mut self, path: &str, dest: &Path) -> Result<(), String>;

    /// ponytail: FTP and HTTP download the object and then keep `max` bytes.
    /// SFTP overrides this and stops the read at `max`.
    fn read_prefix(&mut self, path: &str, max: usize) -> Result<Option<Vec<u8>>, String> {
        let Some(bytes) = self.read(path)? else {
            return Ok(None);
        };
        Ok(Some(bytes.into_iter().take(max).collect()))
    }
}

/// ponytail: a session deeper than this is not fully copied. Raise the constant.
const COPY_DEPTH: u32 = 32;
const FAKE_MTIME_NS: i64 = 1_000_000_000;

struct Fixture {
    files: BTreeMap<String, Vec<u8>>,
    auth: bool,
    copies: u32,
}

fn fixtures() -> &'static Mutex<BTreeMap<String, Fixture>> {
    static FIXTURES: OnceLock<Mutex<BTreeMap<String, Fixture>>> = OnceLock::new();
    FIXTURES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[cfg(test)]
pub(crate) fn install_fixture(canonical: &str, root: &str, files: &[(&str, &[u8])], auth: bool) {
    let mut stored = BTreeMap::new();
    for (relative, bytes) in files {
        let relative = relative.trim_start_matches('/');
        let full = format!("{}/{}", root.trim_end_matches('/'), relative);
        stored.insert(full, bytes.to_vec());
    }
    fixtures()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .insert(
            canonical.to_owned(),
            Fixture {
                files: stored,
                auth,
                copies: 0,
            },
        );
}

#[cfg(test)]
pub(crate) fn fixture_copies(canonical: &str) -> u32 {
    fixtures()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(canonical)
        .map(|fixture| fixture.copies)
        .unwrap_or(0)
}

pub(crate) fn connect(remote: &Remote, password: &str) -> Result<Box<dyn Fs>, String> {
    if let Some(key) = fixture_key(&remote.canonical) {
        let auth = fixtures()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(&key)
            .is_some_and(|fixture| fixture.auth);
        if auth && password.is_empty() {
            return Err(location::auth_error("password required"));
        }
        return Ok(Box::new(MemoryFs { key }));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        net::connect(remote, password)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (remote, password);
        Err("remote locations are not available in this build".into())
    }
}

fn fixture_key(canonical: &str) -> Option<String> {
    let map = fixtures().lock().unwrap_or_else(|err| err.into_inner());
    if map.contains_key(canonical) {
        return Some(canonical.to_owned());
    }
    map.keys()
        .find(|key| canonical.starts_with(&format!("{key}/")))
        .cloned()
}

struct MemoryFs {
    key: String,
}

impl Fs for MemoryFs {
    fn list(&mut self, path: &str) -> Result<Vec<Item>, String> {
        let files = self.files()?;
        Ok(list_memory(&files, path))
    }

    fn read(&mut self, path: &str) -> Result<Option<Vec<u8>>, String> {
        let files = self.files()?;
        Ok(files.get(path.trim_end_matches('/')).cloned())
    }

    fn read_prefix(&mut self, path: &str, max: usize) -> Result<Option<Vec<u8>>, String> {
        let Some(bytes) = self.read(path)? else {
            return Ok(None);
        };
        Ok(Some(bytes.into_iter().take(max).collect()))
    }

    fn stat(&mut self, path: &str) -> Result<Option<Stat>, String> {
        let files = self.files()?;
        Ok(stat_memory(&files, path))
    }

    fn copy_to(&mut self, path: &str, dest: &Path) -> Result<(), String> {
        let files = {
            let mut map = fixtures().lock().unwrap_or_else(|err| err.into_inner());
            let fixture = map
                .get_mut(&self.key)
                .ok_or_else(|| "the remote fixture is gone".to_owned())?;
            fixture.copies += 1;
            fixture.files.clone()
        };
        let path = path.trim_end_matches('/');
        if let Some(bytes) = files.get(path) {
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
            fs::write(dest, bytes).map_err(|err| err.to_string())?;
            return Ok(());
        }
        let prefix = format!("{path}/");
        let mut wrote = false;
        for (file, bytes) in &files {
            let Some(relative) = file.strip_prefix(&prefix) else {
                continue;
            };
            let target = dest.join(relative);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
            fs::write(target, bytes).map_err(|err| err.to_string())?;
            wrote = true;
        }
        if !wrote {
            return Err(format!("nothing to copy from {path}"));
        }
        Ok(())
    }
}

impl MemoryFs {
    fn files(&self) -> Result<BTreeMap<String, Vec<u8>>, String> {
        fixtures()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(&self.key)
            .map(|fixture| fixture.files.clone())
            .ok_or_else(|| "the remote fixture is gone".to_owned())
    }
}

fn list_memory(files: &BTreeMap<String, Vec<u8>>, dir: &str) -> Vec<Item> {
    let dir = dir.trim_end_matches('/');
    let prefix = format!("{dir}/");
    let mut seen = BTreeMap::<String, bool>::new();
    for key in files.keys() {
        let Some(rest) = key.strip_prefix(&prefix) else {
            continue;
        };
        let (name, dir) = match rest.split_once('/') {
            Some((name, _)) => (name, true),
            None => (rest, false),
        };
        if name.is_empty() {
            continue;
        }
        seen.entry(name.to_owned())
            .and_modify(|is_dir| {
                if dir {
                    *is_dir = true;
                }
            })
            .or_insert(dir);
    }
    seen.into_iter()
        .map(|(name, dir)| Item { name, dir })
        .collect()
}

fn stat_memory(files: &BTreeMap<String, Vec<u8>>, path: &str) -> Option<Stat> {
    let path = path.trim_end_matches('/');
    if let Some(bytes) = files.get(path) {
        return Some(Stat {
            dir: false,
            size: bytes.len() as u64,
            mtime_ns: FAKE_MTIME_NS,
        });
    }
    let prefix = format!("{path}/");
    if files.keys().any(|key| key.starts_with(&prefix)) {
        return Some(Stat {
            dir: true,
            size: 0,
            mtime_ns: FAKE_MTIME_NS,
        });
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
mod net {
    use std::fs;
    use std::io::Read;
    use std::net::ToSocketAddrs;
    use std::path::Path;
    use std::sync::Arc;

    use super::{copy_file_to, Fs, Item, Stat, COPY_DEPTH};
    use crate::location::{self, Remote, Scheme};

    pub(super) fn connect(remote: &Remote, password: &str) -> Result<Box<dyn Fs>, String> {
        match remote.scheme {
            Scheme::Sftp => Ok(Box::new(SftpFs::connect(remote, password)?)),
            Scheme::Ftp => Ok(Box::new(FtpFs::connect(remote, password, false)?)),
            Scheme::Ftps => Ok(Box::new(FtpFs::connect(remote, password, true)?)),
            Scheme::Http | Scheme::Https => Ok(Box::new(HttpFs {
                url: remote.canonical.clone(),
                username: remote.username.clone(),
                password: password.to_owned(),
            })),
            Scheme::Smb => Err("an smb URL opens as a Windows UNC path".into()),
        }
    }

    struct SftpFs {
        session: ssh2::Session,
    }

    impl SftpFs {
        fn connect(remote: &Remote, password: &str) -> Result<Self, String> {
            let port = remote.port.unwrap_or(22);
            let address = resolve(&remote.host, port)?;
            let tcp = std::net::TcpStream::connect_timeout(&address, location::connect_timeout())
                .map_err(|err| err.to_string())?;
            let _ = tcp.set_read_timeout(Some(location::connect_timeout()));
            let _ = tcp.set_write_timeout(Some(location::connect_timeout()));
            let mut session = ssh2::Session::new().map_err(|err| err.to_string())?;
            session.set_timeout(20_000);
            session.set_tcp_stream(tcp);
            session.handshake().map_err(|err| err.to_string())?;
            let user = if remote.username.is_empty() {
                "anonymous"
            } else {
                remote.username.as_str()
            };
            session
                .userauth_password(user, password)
                .map_err(|_| location::auth_error("the sftp server refused the password"))?;
            if !session.authenticated() {
                return Err(location::auth_error("the sftp server refused the password"));
            }
            Ok(Self { session })
        }

        fn sftp(&self) -> Result<ssh2::Sftp, String> {
            self.session.sftp().map_err(|err| err.to_string())
        }
    }

    fn read_sftp(
        fs: &mut SftpFs,
        path: &str,
        max: Option<usize>,
    ) -> Result<Option<Vec<u8>>, String> {
        let sftp = fs.sftp()?;
        match sftp.open(Path::new(path)) {
            Ok(mut file) => {
                if let Some(max) = max {
                    let mut bytes = vec![0u8; max];
                    let mut filled = 0;
                    while filled < max {
                        match file.read(&mut bytes[filled..]) {
                            Ok(0) => break,
                            Ok(n) => filled += n,
                            Err(err) => return Err(err.to_string()),
                        }
                    }
                    bytes.truncate(filled);
                    return Ok(Some(bytes));
                }
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)
                    .map_err(|err| err.to_string())?;
                Ok(Some(bytes))
            }
            Err(err) if missing_sftp(&err) => Ok(None),
            Err(err) => Err(map_sftp(err)),
        }
    }

    impl Fs for SftpFs {
        fn list(&mut self, path: &str) -> Result<Vec<Item>, String> {
            let sftp = self.sftp()?;
            let entries = sftp.readdir(Path::new(path)).map_err(map_sftp)?;
            let mut items = Vec::new();
            for (entry, stat) in entries {
                if stat.file_type().is_symlink() {
                    continue;
                }
                let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if name == "." || name == ".." {
                    continue;
                }
                items.push(Item {
                    name: name.to_owned(),
                    dir: stat.is_dir(),
                });
            }
            Ok(items)
        }

        fn read(&mut self, path: &str) -> Result<Option<Vec<u8>>, String> {
            read_sftp(self, path, None)
        }

        fn read_prefix(&mut self, path: &str, max: usize) -> Result<Option<Vec<u8>>, String> {
            read_sftp(self, path, Some(max))
        }

        fn stat(&mut self, path: &str) -> Result<Option<Stat>, String> {
            let sftp = self.sftp()?;
            match sftp.lstat(Path::new(path)) {
                Ok(stat) if stat.file_type().is_symlink() => Ok(None),
                Ok(stat) => Ok(Some(Stat {
                    dir: stat.is_dir(),
                    size: stat.size.unwrap_or(0),
                    mtime_ns: i64::try_from(stat.mtime.unwrap_or(0)).unwrap_or(0) * 1_000_000_000,
                })),
                Err(err) if missing_sftp(&err) => Ok(None),
                Err(err) => Err(map_sftp(err)),
            }
        }

        fn copy_to(&mut self, path: &str, dest: &Path) -> Result<(), String> {
            let stat = self.stat(path)?.ok_or_else(|| format!("missing {path}"))?;
            if stat.dir {
                copy_sftp_dir(self, path, dest, 0)
            } else {
                let bytes = self.read(path)?.ok_or_else(|| format!("missing {path}"))?;
                copy_file_to(dest, &bytes)
            }
        }
    }

    fn copy_sftp_dir(fs: &mut SftpFs, path: &str, dest: &Path, depth: u32) -> Result<(), String> {
        if depth > COPY_DEPTH {
            return Ok(());
        }
        fs::create_dir_all(dest).map_err(|err| err.to_string())?;
        for item in fs.list(path)? {
            let remote = format!("{}/{}", path.trim_end_matches('/'), item.name);
            let local = dest.join(&item.name);
            if item.dir {
                copy_sftp_dir(fs, &remote, &local, depth + 1)?;
            } else if let Some(bytes) = fs.read(&remote)? {
                copy_file_to(&local, &bytes)?;
            }
        }
        Ok(())
    }

    fn missing_sftp(err: &ssh2::Error) -> bool {
        matches!(err.code(), ssh2::ErrorCode::SFTP(2))
    }

    fn map_sftp(err: ssh2::Error) -> String {
        let text = err.to_string();
        let lower = text.to_ascii_lowercase();
        if lower.contains("auth") || lower.contains("password") {
            location::auth_error("the sftp server refused the password")
        } else {
            text
        }
    }

    struct FtpFs {
        client: FtpClient,
    }

    enum FtpClient {
        Plain(suppaftp::FtpStream),
        Tls(suppaftp::RustlsFtpStream),
    }

    impl FtpFs {
        fn connect(remote: &Remote, password: &str, tls: bool) -> Result<Self, String> {
            let port = remote.port.unwrap_or(21);
            let address = resolve(&remote.host, port)?;
            let user = if remote.username.is_empty() {
                "anonymous"
            } else {
                remote.username.as_str()
            };
            let secret = if password.is_empty() && user == "anonymous" {
                "anonymous@"
            } else {
                password
            };
            let client = if tls {
                let config = tls_config()?;
                let stream = suppaftp::RustlsFtpStream::connect_timeout(
                    address,
                    location::connect_timeout(),
                )
                .map_err(map_ftp)?
                .into_secure(
                    suppaftp::RustlsConnector::from(Arc::new(config)),
                    remote.host.trim_matches(['[', ']']),
                )
                .map_err(map_ftp)?;
                let mut stream = stream;
                login_tls(&mut stream, user, secret)?;
                let _ = stream.transfer_type(suppaftp::types::FileType::Binary);
                FtpClient::Tls(stream)
            } else {
                let mut stream =
                    suppaftp::FtpStream::connect_timeout(address, location::connect_timeout())
                        .map_err(map_ftp)?;
                login_plain(&mut stream, user, secret)?;
                let _ = stream.transfer_type(suppaftp::types::FileType::Binary);
                FtpClient::Plain(stream)
            };
            Ok(Self { client })
        }
    }

    fn login_plain(
        ftp: &mut suppaftp::FtpStream,
        user: &str,
        password: &str,
    ) -> Result<(), String> {
        ftp.login(user, password).map_err(map_ftp_auth)
    }

    fn login_tls(
        ftp: &mut suppaftp::RustlsFtpStream,
        user: &str,
        password: &str,
    ) -> Result<(), String> {
        ftp.login(user, password).map_err(map_ftp_auth)
    }

    impl Fs for FtpFs {
        fn list(&mut self, path: &str) -> Result<Vec<Item>, String> {
            let lines = match &mut self.client {
                FtpClient::Plain(ftp) => ftp.list(Some(path)),
                FtpClient::Tls(ftp) => ftp.list(Some(path)),
            }
            .map_err(map_ftp)?;
            let mut items = Vec::new();
            for line in lines {
                let Ok(file) = suppaftp::list::File::try_from(line.as_str()) else {
                    continue;
                };
                if file.is_symlink() {
                    continue;
                }
                let name = file.name().trim();
                if name.is_empty() || name == "." || name == ".." {
                    continue;
                }
                items.push(Item {
                    name: name.to_owned(),
                    dir: file.is_directory(),
                });
            }
            Ok(items)
        }

        fn read(&mut self, path: &str) -> Result<Option<Vec<u8>>, String> {
            let result = match &mut self.client {
                FtpClient::Plain(ftp) => ftp.retr_as_buffer(path),
                FtpClient::Tls(ftp) => ftp.retr_as_buffer(path),
            };
            match result {
                Ok(cursor) => Ok(Some(cursor.into_inner())),
                Err(err) if missing_ftp(&err) => Ok(None),
                Err(err) => Err(map_ftp(err)),
            }
        }

        fn stat(&mut self, path: &str) -> Result<Option<Stat>, String> {
            let listed = match &mut self.client {
                FtpClient::Plain(ftp) => ftp.mlst(Some(path)),
                FtpClient::Tls(ftp) => ftp.mlst(Some(path)),
            };
            match listed {
                Ok(line) => Ok(Some(stat_mlst(&line))),
                Err(err) if missing_ftp(&err) => Ok(None),
                Err(_) => stat_from_list(self, path),
            }
        }

        fn copy_to(&mut self, path: &str, dest: &Path) -> Result<(), String> {
            let stat = self.stat(path)?.ok_or_else(|| format!("missing {path}"))?;
            if stat.dir {
                copy_ftp_dir(self, path, dest, 0)
            } else {
                let bytes = self.read(path)?.ok_or_else(|| format!("missing {path}"))?;
                copy_file_to(dest, &bytes)
            }
        }
    }

    fn copy_ftp_dir(fs: &mut FtpFs, path: &str, dest: &Path, depth: u32) -> Result<(), String> {
        if depth > COPY_DEPTH {
            return Ok(());
        }
        fs::create_dir_all(dest).map_err(|err| err.to_string())?;
        for item in fs.list(path)? {
            let remote = format!("{}/{}", path.trim_end_matches('/'), item.name);
            let local = dest.join(&item.name);
            if item.dir {
                copy_ftp_dir(fs, &remote, &local, depth + 1)?;
            } else if let Some(bytes) = fs.read(&remote)? {
                copy_file_to(&local, &bytes)?;
            }
        }
        Ok(())
    }

    fn stat_from_list(fs: &mut FtpFs, path: &str) -> Result<Option<Stat>, String> {
        let path = path.trim_end_matches('/');
        let Some((parent, name)) = path.rfind('/').map(|index| {
            let parent = if index == 0 { "/" } else { &path[..index] };
            (parent, &path[index + 1..])
        }) else {
            return Ok(None);
        };
        for item in fs.list(parent)? {
            if item.name == name {
                return Ok(Some(Stat {
                    dir: item.dir,
                    size: 0,
                    mtime_ns: 0,
                }));
            }
        }
        Ok(None)
    }

    fn stat_mlst(line: &str) -> Stat {
        let mut dir = false;
        let mut size = 0u64;
        let mut mtime_ns = 0i64;
        for fact in line.split(';') {
            let Some((key, value)) = fact.split_once('=') else {
                continue;
            };
            match key.trim().to_ascii_lowercase().as_str() {
                "type" => {
                    dir = value.eq_ignore_ascii_case("dir")
                        || value.eq_ignore_ascii_case("cdir")
                        || value.eq_ignore_ascii_case("pdir")
                }
                "size" => size = value.trim().parse().unwrap_or(0),
                "modify" => mtime_ns = modify_ns(value.trim()),
                _ => {}
            }
        }
        Stat {
            dir,
            size,
            mtime_ns,
        }
    }

    fn modify_ns(text: &str) -> i64 {
        if text.len() < 14 || !text.bytes().take(14).all(|byte| byte.is_ascii_digit()) {
            return 0;
        }
        let year: i64 = text[0..4].parse().unwrap_or(0);
        let month: i64 = text[4..6].parse().unwrap_or(1);
        let day: i64 = text[6..8].parse().unwrap_or(1);
        let hour: i64 = text[8..10].parse().unwrap_or(0);
        let minute: i64 = text[10..12].parse().unwrap_or(0);
        let second: i64 = text[12..14].parse().unwrap_or(0);
        let days = days_from_civil(year, month, day);
        (days * 86_400 + hour * 3_600 + minute * 60 + second) * 1_000_000_000
    }

    fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
        let year = if month <= 2 { year - 1 } else { year };
        let era = if year >= 0 { year } else { year - 399 } / 400;
        let yoe = year - era * 400;
        let month_index = if month > 2 { month - 3 } else { month + 9 };
        let doy = (153 * month_index + 2) / 5 + day - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    fn missing_ftp(err: &suppaftp::FtpError) -> bool {
        match err {
            suppaftp::FtpError::UnexpectedResponse(response) => {
                matches!(response.status as u32, 450 | 550)
            }
            _ => false,
        }
    }

    fn map_ftp_auth(err: suppaftp::FtpError) -> String {
        if ftp_auth(&err) {
            location::auth_error("the ftp server refused the password")
        } else {
            err.to_string()
        }
    }

    fn map_ftp(err: suppaftp::FtpError) -> String {
        if ftp_auth(&err) {
            location::auth_error("the ftp server refused the password")
        } else {
            err.to_string()
        }
    }

    fn ftp_auth(err: &suppaftp::FtpError) -> bool {
        match err {
            suppaftp::FtpError::UnexpectedResponse(response) => {
                matches!(response.status as u32, 332 | 421 | 530)
            }
            _ => {
                let text = err.to_string().to_ascii_lowercase();
                text.contains("auth") || text.contains("login") || text.contains("password")
            }
        }
    }

    fn tls_config() -> Result<suppaftp::rustls::ClientConfig, String> {
        let mut roots = suppaftp::rustls::RootCertStore::empty();
        let loaded = rustls_native_certs::load_native_certs();
        if loaded.certs.is_empty() {
            let detail = loaded
                .errors
                .first()
                .map(|err| err.to_string())
                .unwrap_or_else(|| {
                    "the system certificate store has no certificates for FTPS".into()
                });
            return Err(detail);
        }
        for cert in loaded.certs {
            roots.add(cert).map_err(|err| err.to_string())?;
        }
        let provider = Arc::new(suppaftp::rustls::crypto::ring::default_provider());
        Ok(
            suppaftp::rustls::ClientConfig::builder_with_provider(provider)
                .with_protocol_versions(suppaftp::rustls::DEFAULT_VERSIONS)
                .map_err(|err| err.to_string())?
                .with_root_certificates(roots)
                .with_no_client_auth(),
        )
    }

    struct HttpFs {
        url: String,
        username: String,
        password: String,
    }

    impl Fs for HttpFs {
        fn list(&mut self, _path: &str) -> Result<Vec<Item>, String> {
            Err("an http location is one archive, not a folder".into())
        }

        fn read(&mut self, _path: &str) -> Result<Option<Vec<u8>>, String> {
            let bytes = http_get(&self.url, &self.username, &self.password)?;
            Ok(Some(bytes))
        }

        fn stat(&mut self, _path: &str) -> Result<Option<Stat>, String> {
            Ok(Some(http_stat(&self.url, &self.username, &self.password)?))
        }

        fn copy_to(&mut self, _path: &str, dest: &Path) -> Result<(), String> {
            let bytes = http_get(&self.url, &self.username, &self.password)?;
            copy_file_to(dest, &bytes)
        }
    }

    fn http_agent() -> Result<ureq::Agent, String> {
        let connector = native_tls::TlsConnector::new().map_err(|err| err.to_string())?;
        Ok(ureq::AgentBuilder::new()
            .timeout_connect(location::connect_timeout())
            .timeout_read(location::connect_timeout())
            .timeout_write(location::connect_timeout())
            .tls_connector(std::sync::Arc::new(connector))
            .build())
    }

    fn http_get(url: &str, username: &str, password: &str) -> Result<Vec<u8>, String> {
        let response = authed(http_agent()?.get(url), username, password)
            .call()
            .map_err(map_http)?;
        let mut bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .map_err(|err| err.to_string())?;
        Ok(bytes)
    }

    fn http_stat(url: &str, username: &str, password: &str) -> Result<Stat, String> {
        // ponytail: a server that rejects HEAD keeps a zero stamp, so the cached
        // archive is reused until the process forgets it. A Content-Length change
        // still downloads again.
        let response = authed(http_agent()?.request("HEAD", url), username, password).call();
        let response = match response {
            Ok(response) => response,
            Err(ureq::Error::Status(405, _)) => {
                return Ok(Stat {
                    dir: false,
                    size: 0,
                    mtime_ns: 0,
                })
            }
            Err(err) => return Err(map_http(err)),
        };
        let size = response
            .header("content-length")
            .and_then(|text| text.parse().ok())
            .unwrap_or(0);
        Ok(Stat {
            dir: false,
            size,
            mtime_ns: 0,
        })
    }

    fn authed(request: ureq::Request, username: &str, password: &str) -> ureq::Request {
        if username.is_empty() && password.is_empty() {
            return request;
        }
        request.set("Authorization", &basic_auth(username, password))
    }

    fn basic_auth(username: &str, password: &str) -> String {
        let raw = format!("{username}:{password}");
        format!("Basic {}", base64(raw.as_bytes()))
    }

    fn base64(bytes: &[u8]) -> String {
        const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        let mut index = 0;
        while index + 3 <= bytes.len() {
            let value = ((bytes[index] as u32) << 16)
                | ((bytes[index + 1] as u32) << 8)
                | bytes[index + 2] as u32;
            out.push(TABLE[((value >> 18) & 63) as usize] as char);
            out.push(TABLE[((value >> 12) & 63) as usize] as char);
            out.push(TABLE[((value >> 6) & 63) as usize] as char);
            out.push(TABLE[(value & 63) as usize] as char);
            index += 3;
        }
        let rest = bytes.len() - index;
        if rest == 1 {
            let value = (bytes[index] as u32) << 16;
            out.push(TABLE[((value >> 18) & 63) as usize] as char);
            out.push(TABLE[((value >> 12) & 63) as usize] as char);
            out.push('=');
            out.push('=');
        } else if rest == 2 {
            let value = ((bytes[index] as u32) << 16) | ((bytes[index + 1] as u32) << 8);
            out.push(TABLE[((value >> 18) & 63) as usize] as char);
            out.push(TABLE[((value >> 12) & 63) as usize] as char);
            out.push(TABLE[((value >> 6) & 63) as usize] as char);
            out.push('=');
        }
        out
    }

    fn map_http(err: ureq::Error) -> String {
        match err {
            ureq::Error::Status(code, _) if code == 401 || code == 403 => {
                location::auth_error("the archive URL refused the password")
            }
            other => other.to_string(),
        }
    }

    fn resolve(host: &str, port: u16) -> Result<std::net::SocketAddr, String> {
        let host = host.trim_matches(['[', ']']);
        let text = if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        };
        text.to_socket_addrs()
            .map_err(|err| err.to_string())?
            .next()
            .ok_or_else(|| format!("could not resolve {text}"))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn listing_facts_and_base64_cover_the_helpers() {
            let dir = stat_mlst("type=dir;size=4;modify=19700101000000; name");
            assert!(dir.dir);
            assert_eq!(dir.size, 4);
            assert_eq!(dir.mtime_ns, 0);
            let file = stat_mlst("type=file;size=12;modify=not-a-time; junk");
            assert!(!file.dir);
            assert_eq!(file.size, 12);
            assert_eq!(modify_ns("short"), 0);
            assert_eq!(
                modify_ns("20200102150403"),
                days_from_civil(2020, 1, 2) * 86_400 * 1_000_000_000
                    + (15 * 3_600 + 4 * 60 + 3) * 1_000_000_000
            );
            assert_eq!(days_from_civil(1970, 1, 1), 0);
            assert!(days_from_civil(1969, 12, 31) < 0);
            assert_eq!(resolve("127.0.0.1", 9).unwrap().port(), 9);
            assert_eq!(resolve("[::1]", 9).unwrap().port(), 9);
            assert!(tls_config().is_ok());
            assert_eq!(base64(b"a"), "YQ==");
            assert_eq!(base64(b"ab"), "YWI=");
            assert_eq!(base64(b"abc"), "YWJj");
            assert!(basic_auth("ada", "secret").starts_with("Basic "));
        }
    }
}

fn copy_file_to(dest: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let mut file = File::create(dest).map_err(|err| err.to_string())?;
    file.write_all(bytes).map_err(|err| err.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::Scheme;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn remote(
        canonical: &str,
        scheme: Scheme,
        host: &str,
        port: Option<u16>,
    ) -> crate::location::Remote {
        crate::location::Remote {
            canonical: canonical.to_owned(),
            scheme,
            username: String::new(),
            password: None,
            host: host.to_owned(),
            port,
            path: "/root".into(),
        }
    }

    fn serve(status: &'static [u8]) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for _ in 0..8 {
                let Ok((mut sock, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf);
                let _ = sock.write_all(status);
            }
        });
        port
    }

    #[test]
    fn a_memory_tree_lists_reads_and_copies() {
        let canonical = format!("fixture://memory/{}", std::process::id());
        install_fixture(
            &canonical,
            "/root",
            &[("a.txt", b"hello"), ("dir/b.txt", b"yo")],
            true,
        );
        let spec = remote(&canonical, Scheme::Http, "", None);
        assert!(connect(&spec, "").is_err());
        let mut fs = connect(&spec, "secret").unwrap();
        let names = fs.list("/root").unwrap();
        assert!(names.iter().any(|item| item.name == "a.txt" && !item.dir));
        assert!(names.iter().any(|item| item.name == "dir" && item.dir));
        assert_eq!(fs.read("/root/a.txt").unwrap().unwrap(), b"hello");
        assert!(fs.read("/root/missing").unwrap().is_none());
        assert_eq!(fs.read_prefix("/root/a.txt", 2).unwrap().unwrap(), b"he");
        assert!(fs.read_prefix("/root/missing", 2).unwrap().is_none());
        let file = fs.stat("/root/a.txt").unwrap().unwrap();
        assert!(!file.dir && file.size == 5);
        assert!(fs.stat("/root").unwrap().unwrap().dir);
        assert!(fs.stat("/root/missing").unwrap().is_none());
        let dest = std::env::temp_dir().join(format!("scan-kit-remote-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dest);
        fs.copy_to("/root/a.txt", &dest.join("a.txt")).unwrap();
        assert_eq!(fs::read(dest.join("a.txt")).unwrap(), b"hello");
        fs.copy_to("/root", &dest.join("tree")).unwrap();
        assert_eq!(fs::read(dest.join("tree/dir/b.txt")).unwrap(), b"yo");
        assert!(fs.copy_to("/root/absent", &dest.join("absent")).is_err());
        assert!(fixture_copies(&canonical) >= 2);
        let mut child = spec.clone();
        child.canonical = format!("{canonical}/nested");
        assert!(connect(&child, "secret").is_ok());
        let smb = remote("smb://share", Scheme::Smb, "share", None);
        match connect(&smb, "") {
            Err(message) => assert!(message.contains("UNC")),
            Ok(_) => panic!("an smb url is a local path"),
        }
        let _ = fs::remove_dir_all(&dest);
    }

    #[test]
    fn an_http_archive_reads_stats_and_copies() {
        let ok = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndata";
        let port = serve(ok);
        let url = format!("http://127.0.0.1:{port}/archive.bin");
        let mut spec = remote(&url, Scheme::Http, "127.0.0.1", Some(port));
        spec.username = "ada".into();
        let mut fs = connect(&spec, "secret").unwrap();
        match fs.list("/") {
            Err(message) => assert!(message.contains("one archive")),
            Ok(_) => panic!("http is one archive"),
        }
        assert_eq!(fs.read("/").unwrap().unwrap(), b"data");
        assert_eq!(fs.read_prefix("/", 2).unwrap().unwrap(), b"da");
        let stat = fs.stat("/").unwrap().unwrap();
        assert!(!stat.dir && stat.size == 4);
        let dest = std::env::temp_dir().join(format!("scan-kit-http-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dest);
        fs.copy_to("/", &dest.join("nested/archive.bin")).unwrap();
        assert_eq!(fs::read(dest.join("nested/archive.bin")).unwrap(), b"data");
        let denied = serve(b"HTTP/1.1 401 NO\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        let denied_url = format!("http://127.0.0.1:{denied}/archive.bin");
        let denied_spec = remote(&denied_url, Scheme::Https, "127.0.0.1", Some(denied));
        let mut denied_fs = connect(&denied_spec, "").unwrap();
        assert!(denied_fs.read("/").unwrap_err().contains("password"));
        let head = serve(b"HTTP/1.1 405 NO\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        let head_url = format!("http://127.0.0.1:{head}/archive.bin");
        let head_spec = remote(&head_url, Scheme::Http, "127.0.0.1", Some(head));
        let mut head_fs = connect(&head_spec, "").unwrap();
        let missing = head_fs.stat("/").unwrap().unwrap();
        assert_eq!(missing.size, 0);
        let _ = fs::remove_dir_all(&dest);
    }
}
