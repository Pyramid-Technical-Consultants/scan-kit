---
name: release-version
description: Bump the scan-kit version and cut a release. Covers updating __version__, committing, tagging vX.Y.Z, and pushing to trigger the GitHub Actions build/release. Use when the user asks to bump the version, prepare or cut a release, or tag a new version.
---

# Releasing scan-kit

## Single source of truth

The version lives in **one** place — `scan_kit/__init__.py`:

```python
__version__ = "1.4.0"
```

`pyproject.toml` reads it dynamically (`version = {attr = "scan_kit.__version__"}`), and the launcher window title (`Scan Kit v{__version__}`) picks it up automatically. **Never** hard-code the version anywhere else.

## Choosing the bump (semver)

- **Patch** (`x.y.Z`): bug fixes only, no behavior/UI changes.
- **Minor** (`x.Y.0`): new user-facing features or UI changes, backward compatible.
- **Major** (`X.0.0`): breaking changes to data formats, CLI, or settings.

If unsure, ask the user which bump they want.

## Release workflow

1. **Bump** `__version__` in `scan_kit/__init__.py` (this is the only file to edit).
2. **Commit** with a message matching the repo convention `Release vX.Y.Z`:

```bash
git add scan_kit/__init__.py
git commit -m "Release vX.Y.Z"
```

3. **Tag** the commit (the tag name must be `v` + the version, e.g. `v1.4.0`):

```bash
git tag vX.Y.Z
```

4. **Push** the branch and the tag. Pushing the `v*` tag is what triggers the release (see [How the tag triggers a release](#how-the-tag-triggers-a-release-ci-mechanics)):

```bash
git push
git push origin vX.Y.Z
```

`git push` alone does **not** push tags — push the tag explicitly as above, or use `git push --follow-tags` to send the branch and its annotated tags together.

> On PowerShell, run each command separately — do not chain with `&&`.

## How the tag triggers a release (CI mechanics)

The release is driven entirely by the **pushed git tag**, via `.github/workflows/build.yml`:

```yaml
on:
  push:
    branches: [main]
    tags: ["v*"]
...
  release:
    needs: build
    if: startsWith(github.ref, 'refs/tags/v')
```

What this means in practice:

- The `build` job runs on every push to `main`, every PR, and every `v*` tag — but it only **builds** artifacts, it does **not** release.
- The `release` job runs **only** when the pushed ref is a tag starting with `v` (`refs/tags/v*`). This is the single gate that publishes the GitHub Release and attaches the Windows/Linux binaries.
- Therefore a release happens **only** when you push a tag whose name starts with `v`. Pushing the commit alone, or a tag named without the `v` prefix (e.g. `1.4.0`), will build but **never release**.

Requirements for the trigger to fire correctly:

1. **Tag name starts with `v`** and matches the version, e.g. `v1.4.0`. The glob is `v*`, so `v1.4.0` matches; `1.4.0` does not.
2. **The tag is pushed to the remote.** `git push` does *not* push tags by default — you must push the tag explicitly (`git push origin v1.4.0`) or use `git push --follow-tags`.
3. **The tagged commit already bumped `__version__`.** CI builds the code *at the tag*, so the binary's version comes from whatever the tag points at. Bump first, then tag that commit. The release job fails if the tag and `__version__` disagree.

### Release asset filenames

| Build type | Windows | Linux |
|------------|---------|-------|
| Tagged release | `scan-kit-windows-X.Y.Z.exe` | `scan-kit-linux-amd64-X.Y.Z` |
| Non-tagged CI (PR / `main`) | `scan-kit-windows-X.Y.Z-rc.exe` | `scan-kit-linux-amd64-X.Y.Z-rc` |

`X.Y.Z` comes from `__version__` for `-rc` builds and from the git tag (without `v`) for releases. No changelog file is maintained; GitHub auto-generates release notes.

## Fixing or re-doing a release

If a tag was pushed at the wrong commit or with the wrong name, delete it locally and on the remote, then re-tag and push:

```bash
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z
git tag vX.Y.Z <correct-commit>
git push origin vX.Y.Z
```

Deleting a tag does not delete an already-published GitHub Release — remove that in the GitHub UI (or with `gh release delete vX.Y.Z`) if it was created.

## Verification checklist

- [ ] `__version__` updated and is the only changed line for the bump.
- [ ] Commit message is `Release vX.Y.Z`.
- [ ] Tag name exactly matches `vX.Y.Z` (same as `__version__`, prefixed with `v`).
- [ ] Tag points at the release commit and was pushed.
- [ ] CI release job passes the tag/`__version__` sanity check and publishes both platform binaries.

## Notes

- Only create the tag/push when the user explicitly wants to release. If they just want the number bumped (e.g. inside an in-progress PR), do step 1 only and skip tagging.
- If this is part of a PR that isn't merged yet, prefer bumping on the PR branch; the tag is usually created after merge to `main`.
- PR and `main` pushes produce `-rc` artifacts in the Actions run for pre-release testing; they do not create a GitHub Release.
