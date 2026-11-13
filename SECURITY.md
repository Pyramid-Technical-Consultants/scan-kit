# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately, either through GitHub's **Report a vulnerability** button on this repository's Security tab or by email to support@ptcusa.com. Do not open a public issue.

Include the Scan Kit version (Help → About), the operating system, and the steps or input that show the problem. We acknowledge a report within five business days, keep you informed while we work on it, and credit you in the release notes unless you ask us not to.

## Supported versions

Fixes go into the latest release. Older releases do not receive patches.

## What is in scope

The desktop application and installers, the MCP server, and the session, DICOM, configuration, and network code they share. Scan Kit talks to RCI controllers and file servers on the clinic network, and reads files that come from those systems, so issues in parsing, remote access, or device control are in scope.
