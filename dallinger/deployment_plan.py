"""Build deterministic, content-addressed deployment manifests.

This module turns a literal ``deploy.toml`` policy into an ordered plan of
regular experiment-root files. The ``[exclude]`` table has ``paths``
(root-relative prefixes), ``names`` (basenames in every directory), and
``suffixes`` (``.db``-style endings in every directory).
Generated outputs, framework providers, and backend materialization live
outside this module.

Traversal uses ordinary ``lstat`` / ``scandir`` checks (no symlink following).
Selected symlinks and special files are rejected.

A plan freezes one auditable deployment *generation*. Every planned file is
hashed at planning time, and :attr:`DeploymentPlan.generation` identifies the
complete content set (policy + destinations + digests + modes). Reads of the
working tree are gated by :func:`verify_plan_sources`, and materialization
streams each source through the same digest while copying, so a replacement,
deletion, type change, mode change, or newly selected file terminates the
generation before any publishable archive or image is produced. The fully
assembled, isolated tree is itself recorded in an :class:`AssemblyManifest`
that archives and Docker contexts are rechecked against.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import stat
import tomllib
import unicodedata
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, TypeVar

POLICY_FILENAME = "deploy.toml"
SCHEMA_VERSION = 1
DIGEST_PREFIX = "sha256:"
HASH_CHUNK_SIZE = 1024 * 1024
ASSEMBLY_MANIFEST_NAME = ".dallinger-assembly-manifest.json"
ASSEMBLY_MANIFEST_VERSION = 1
# Docker label carrying the frozen content generation of a built image.
GENERATION_LABEL = "org.dallinger.deployment-generation"

_POLICY_KEYS = frozenset({"version", "exclude"})
_EXCLUDE_KEYS = frozenset({"paths", "names", "suffixes"})
_VCS_METADATA_NAMES = frozenset({".git", ".hg", ".svn", ".bzr"})
_GENERATED_ROOT_DESTINATION_NAMES = frozenset(
    {
        "config.txt",
        "constraints.txt",
        "experiment_id.txt",
        "requirements.txt",
        "runtime.txt",
    }
)
_GLOB_CHARACTERS = frozenset("*?[]{}")

_ErrorType = TypeVar("_ErrorType", bound=ValueError)


class DeploymentPolicyError(ValueError):
    """Raised when ``deploy.toml`` is missing, unsafe, or invalid."""


class DeploymentPlanError(ValueError):
    """Raised when an experiment tree cannot produce a safe deployment plan."""


class DeploymentGenerationError(DeploymentPlanError):
    """Raised when working-tree inputs drift from the frozen generation.

    The deployment must not publish an archive or image for a generation in
    this state: replanning starts a new generation with a new isolated
    artifact instead of silently overwriting the original attempt.
    """


@dataclass(frozen=True)
class DeploymentPolicy:
    """The validated, normalized contents of a version 1 ``deploy.toml``."""

    version: int
    exclude_paths: tuple[str, ...]
    exclude_names: tuple[str, ...]
    exclude_suffixes: tuple[str, ...]


@dataclass(frozen=True)
class DeploymentPlanEntry:
    """One regular experiment file selected for deployment."""

    source: Path
    destination: str
    size: int
    mode: int
    executable: bool
    digest: str


@dataclass(frozen=True)
class DeploymentDirectoryLinkCandidate:
    """One fully selected source directory eligible for a development link."""

    source: Path
    destination: str
    entry_start: int
    entry_stop: int

    @property
    def entry_count(self) -> int:
        """Return the number of planned files covered by this directory."""
        return self.entry_stop - self.entry_start


@dataclass(frozen=True)
class DeploymentPlan:
    """An immutable, deterministically ordered experiment-root plan."""

    root: Path
    policy: DeploymentPolicy
    entries: tuple[DeploymentPlanEntry, ...]
    destinations: frozenset[str]
    total_size: int
    directory_link_candidates: tuple[DeploymentDirectoryLinkCandidate, ...]
    policy_digest: str
    generation: str

    def __contains__(self, destination: object) -> bool:
        """Return whether a normalized destination is present in the plan."""
        return isinstance(destination, str) and destination in self.destinations


@dataclass(frozen=True)
class _PolicySnapshot:
    policy: DeploymentPolicy
    size: int
    mode: int
    digest: str


@dataclass(frozen=True)
class _TraversedDirectoryCandidate:
    source: Path
    destination: str
    entry_count: int


@dataclass(frozen=True)
class AssemblyFileRecord:
    """One regular file recorded in a frozen assembly manifest."""

    destination: str
    digest: str
    size: int
    mode: int


@dataclass(frozen=True)
class AssemblyManifest:
    """Content identity of one fully materialized, isolated deployment tree."""

    generation: str
    files: tuple[AssemblyFileRecord, ...]
    root_digest: str

    def to_json(self) -> str:
        """Serialize the manifest to deterministic, auditable JSON."""
        return json.dumps(
            {
                "version": ASSEMBLY_MANIFEST_VERSION,
                "generation": self.generation,
                "root_digest": self.root_digest,
                "files": [
                    {
                        "destination": record.destination,
                        "digest": record.digest,
                        "size": record.size,
                        "mode": record.mode,
                    }
                    for record in self.files
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> "AssemblyManifest":
        """Parse one manifest written by :meth:`to_json`."""
        try:
            raw = json.loads(text)
        except (TypeError, json.JSONDecodeError) as error:
            raise DeploymentPlanError(f"Invalid assembly manifest JSON: {error}") from error
        if not isinstance(raw, dict):
            raise DeploymentPlanError("Assembly manifest must be a JSON object.")
        if raw.get("version") != ASSEMBLY_MANIFEST_VERSION:
            raise DeploymentPlanError(
                f"Unsupported assembly manifest version: {raw.get('version')!r}."
            )
        generation = raw.get("generation")
        if not isinstance(generation, str) or not generation:
            raise DeploymentPlanError("Assembly manifest is missing its generation.")
        raw_files = raw.get("files")
        if not isinstance(raw_files, list):
            raise DeploymentPlanError("Assembly manifest files must be a list.")
        records: list[AssemblyFileRecord] = []
        for item in raw_files:
            if not isinstance(item, dict):
                raise DeploymentPlanError("Each assembly manifest file must be an object.")
            destination = item.get("destination")
            digest = item.get("digest")
            size = item.get("size")
            mode = item.get("mode")
            if not isinstance(destination, str) or not destination:
                raise DeploymentPlanError("Assembly manifest file is missing a destination.")
            if not isinstance(digest, str) or not digest.startswith(DIGEST_PREFIX):
                raise DeploymentPlanError(
                    f"Assembly manifest file {destination!r} has an invalid digest."
                )
            if type(size) is not int or size < 0:
                raise DeploymentPlanError(
                    f"Assembly manifest file {destination!r} has an invalid size."
                )
            if type(mode) is not int or not 0 <= mode <= 0o7777:
                raise DeploymentPlanError(
                    f"Assembly manifest file {destination!r} has an invalid mode."
                )
            records.append(
                AssemblyFileRecord(
                    destination=destination,
                    digest=digest,
                    size=size,
                    mode=mode,
                )
            )
        records = tuple(sorted(records, key=lambda record: record.destination))
        root_digest = assembly_root_digest(generation, records)
        recorded_root = raw.get("root_digest")
        if not isinstance(recorded_root, str) or recorded_root != root_digest:
            raise DeploymentPlanError(
                "Assembly manifest root digest does not match its recorded files."
            )
        return cls(generation=generation, files=records, root_digest=root_digest)


def parse_deployment_policy(path: str | os.PathLike[str]) -> DeploymentPolicy:
    """Load and validate a version 1 literal deployment policy."""
    _require_posix_support(DeploymentPolicyError)
    policy_path = Path(os.path.abspath(os.fspath(path)))
    return _read_policy_snapshot(policy_path).policy


def validate_explicit_provider_destination(
    destination: str | os.PathLike[str],
) -> str:
    """Validate one root-relative destination supplied by ``extra_files()``."""
    value = os.fspath(destination)
    if not value or value.startswith("/"):
        raise DeploymentPlanError(
            f"Explicit file provider destination must be root-relative: {value!r}."
        )
    if "\\" in value:
        raise DeploymentPlanError(
            f"Explicit file provider destinations must use POSIX separators: {value!r}."
        )

    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise DeploymentPlanError(
            "Explicit file provider destination contains an unsafe path component: "
            f"{value!r}."
        )
    normalized_parts = tuple(
        _normalize_path_component(
            component,
            context=f"explicit file provider destination {value!r}",
            error_type=DeploymentPlanError,
        )
        for component in components
    )
    normalized = "/".join(normalized_parts)
    reserved_kind = _reserved_kind(normalized_parts)
    if (
        reserved_kind is None
        and normalized_parts[0] in _GENERATED_ROOT_DESTINATION_NAMES
    ):
        reserved_kind = "generated"
    if reserved_kind is not None:
        raise DeploymentPlanError(
            "Explicit file provider cannot target reserved deployment destination "
            f"{normalized!r} ({reserved_kind})."
        )
    return normalized


def _parse_policy(raw_policy: dict) -> DeploymentPolicy:
    unknown_keys = set(raw_policy) - _POLICY_KEYS
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise DeploymentPolicyError(f"Unknown deployment policy key(s): {keys}.")

    missing = {"version", "exclude"} - set(raw_policy)
    if missing:
        keys = ", ".join(sorted(missing))
        raise DeploymentPolicyError(f"Missing deployment policy key(s): {keys}.")

    version = raw_policy["version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise DeploymentPolicyError(
            f"Deployment policy version must be the integer {SCHEMA_VERSION}."
        )

    exclude_table = raw_policy["exclude"]
    if not isinstance(exclude_table, dict):
        raise DeploymentPolicyError(
            "Deployment policy exclude must be a table with paths, names, and suffixes."
        )
    unknown_exclude_keys = set(exclude_table) - _EXCLUDE_KEYS
    if unknown_exclude_keys:
        keys = ", ".join(sorted(unknown_exclude_keys))
        raise DeploymentPolicyError(f"Unknown exclude key(s): {keys}.")

    paths = _normalized_unique_strings(
        _optional_string_list(exclude_table, "paths"),
        validator=_validate_policy_path,
        duplicate_label="exclude.paths entry",
    )
    names = _normalized_unique_strings(
        _optional_string_list(exclude_table, "names"),
        validator=_validate_exclude_name,
        duplicate_label="exclude.names entry",
    )
    suffixes = _normalized_unique_strings(
        _optional_string_list(exclude_table, "suffixes"),
        validator=_validate_exclude_suffix,
        duplicate_label="exclude.suffixes entry",
    )

    return DeploymentPolicy(
        version=version,
        exclude_paths=tuple(sorted(paths)),
        exclude_names=tuple(sorted(names)),
        exclude_suffixes=tuple(sorted(suffixes)),
    )


def build_deployment_plan(
    experiment_root: str | os.PathLike[str],
) -> DeploymentPlan:
    """Build a deployment plan containing regular experiment-root files."""
    _require_posix_support(DeploymentPlanError)
    root = Path(os.path.abspath(os.fspath(experiment_root)))
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise DeploymentPlanError(
            f"Cannot inspect experiment root {root}: {error}"
        ) from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise DeploymentPlanError(f"Experiment root is not a directory: {root}.")

    entries: list[DeploymentPlanEntry] = []
    normalized_sources: dict[str, Path] = {}
    traversed_directory_candidates: list[_TraversedDirectoryCandidate] = []

    def register_destination(destination: str, source: Path) -> None:
        existing = normalized_sources.get(destination)
        if existing is not None and existing != source:
            raise DeploymentPlanError(
                "Source paths normalize to the same deployment destination "
                f"{destination!r}: {existing} and {source}."
            )
        normalized_sources[destination] = source

    def walk(
        directory: Path,
        destination_parts: tuple[str, ...],
    ) -> bool:
        """Traverse one directory and report whether all descendants are selected."""
        all_descendants_selected = not _has_exclusion_at_or_below(
            destination_parts, exclusions
        )
        try:
            scanner = os.scandir(directory)
        except OSError as error:
            raise DeploymentPlanError(
                f"Cannot scan experiment directory {directory}: {error}"
            ) from error

        with scanner:
            for child in scanner:
                raw_name = child.name
                source = directory / raw_name
                name = _normalize_source_component(raw_name, source)
                child_destination_parts = (*destination_parts, name)
                destination = "/".join(child_destination_parts)

                if _is_excluded(child_destination_parts, exclusions):
                    all_descendants_selected = False
                    continue
                if _is_omitted_anywhere(name, anywhere_names, anywhere_suffixes):
                    all_descendants_selected = False
                    continue

                reserved_kind = _reserved_kind(child_destination_parts)
                if reserved_kind in {"backend-ignore", "policy", "configuration"}:
                    all_descendants_selected = False
                    continue
                if reserved_kind == "vcs":
                    if destination_parts:
                        raise DeploymentPlanError(
                            "Nested repository or submodule metadata is unsupported: "
                            f"{source}. Exclude the complete nested repository or move "
                            "it outside the experiment root."
                        )
                    all_descendants_selected = False
                    continue
                if reserved_kind is not None:
                    all_descendants_selected = False
                    continue

                register_destination(destination, source)
                try:
                    source_stat = child.stat(follow_symlinks=False)
                except OSError as error:
                    raise DeploymentPlanError(
                        f"Cannot inspect deployment source {source}: {error}"
                    ) from error

                if stat.S_ISLNK(source_stat.st_mode):
                    raise DeploymentPlanError(
                        f"Symbolic links are unsupported deployment sources: {source}. "
                        "Exclude the link or replace it with a regular file or directory."
                    )
                if stat.S_ISDIR(source_stat.st_mode):
                    child_entry_start = len(entries)
                    child_all_selected = walk(source, child_destination_parts)
                    child_entry_count = len(entries) - child_entry_start
                    if child_all_selected and child_entry_count:
                        traversed_directory_candidates.append(
                            _TraversedDirectoryCandidate(
                                source=source,
                                destination=destination,
                                entry_count=child_entry_count,
                            )
                        )
                    if not child_all_selected:
                        all_descendants_selected = False
                    continue
                if not stat.S_ISREG(source_stat.st_mode):
                    source_type = _source_type(source_stat.st_mode)
                    raise DeploymentPlanError(
                        f"Unsupported {source_type} deployment source: {source}. "
                        "Only regular files and directories are supported."
                    )

                mode = stat.S_IMODE(source_stat.st_mode)
                digest, read_size = _hash_regular_file(source)
                if read_size != source_stat.st_size:
                    raise DeploymentPlanError(
                        f"Deployment source changed while reading {source}: "
                        f"stat reported {source_stat.st_size} bytes but read "
                        f"{read_size}. Plan this deployment again."
                    )
                entries.append(
                    DeploymentPlanEntry(
                        source=source,
                        destination=destination,
                        size=source_stat.st_size,
                        mode=mode,
                        executable=bool(mode & 0o111),
                        digest=digest,
                    )
                )
        return all_descendants_selected

    policy_snapshot = _read_policy_snapshot(root / POLICY_FILENAME)
    policy = policy_snapshot.policy
    exclusions = frozenset(policy.exclude_paths)
    anywhere_names = frozenset(policy.exclude_names)
    anywhere_suffixes = policy.exclude_suffixes
    policy_source = root / POLICY_FILENAME
    register_destination(POLICY_FILENAME, policy_source)
    entries.append(
        DeploymentPlanEntry(
            source=policy_source,
            destination=POLICY_FILENAME,
            size=policy_snapshot.size,
            mode=policy_snapshot.mode,
            executable=bool(policy_snapshot.mode & 0o111),
            digest=policy_snapshot.digest,
        )
    )
    walk(root, ())

    ordered_entries = tuple(sorted(entries, key=lambda entry: entry.destination))
    ordered_destinations = tuple(entry.destination for entry in ordered_entries)
    destinations = frozenset(ordered_destinations)
    directory_link_candidates = tuple(
        sorted(
            (
                _finalize_directory_candidate(candidate, ordered_destinations)
                for candidate in traversed_directory_candidates
            ),
            key=lambda candidate: candidate.destination,
        )
    )
    total_size = sum(entry.size for entry in ordered_entries)
    generation = _plan_generation(policy_snapshot.digest, ordered_entries)
    return DeploymentPlan(
        root=root,
        policy=policy,
        entries=ordered_entries,
        destinations=destinations,
        total_size=total_size,
        directory_link_candidates=directory_link_candidates,
        policy_digest=policy_snapshot.digest,
        generation=generation,
    )


def materialize_deployment_plan_entry(
    plan: DeploymentPlan,
    entry: DeploymentPlanEntry,
    destination: str | os.PathLike[str],
) -> None:
    """Copy one planned file into a trusted destination without overwriting.

    The source is streamed through the frozen SHA-256 digest while copying,
    so the bytes landing in the isolated tree are certified to be exactly
    the generation planned at :func:`build_deployment_plan` time. A source
    replaced, truncated, or still changing at copy time raises
    :class:`DeploymentGenerationError` and leaves no target behind.
    """
    target = Path(os.path.abspath(os.fspath(destination)))
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_source_under_root(plan.root, entry.source)
    if target.exists() or target.is_symlink():
        raise DeploymentPlanError(
            f"Cannot materialize deployment entry {entry.destination!r}: "
            f"destination already exists: {target}."
        )
    try:
        source_stat = entry.source.lstat()
    except OSError as error:
        raise DeploymentPlanError(
            f"Cannot inspect deployment source {entry.source}: {error}"
        ) from error
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise DeploymentPlanError(
            f"Deployment source is not a regular file: {entry.source}."
        )

    hasher = hashlib.sha256()
    copied = 0
    try:
        with entry.source.open("rb") as source_file, target.open("wb") as target_file:
            while True:
                chunk = source_file.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                copied += len(chunk)
                target_file.write(chunk)
        digest = _digest_from_hash(hasher)
        mismatches: list[str] = []
        if copied != entry.size:
            mismatches.append(
                f"size {entry.size} bytes at planning but {copied} at materialization"
            )
        if digest != entry.digest:
            mismatches.append("its content digest differs from the frozen generation")
        if mismatches:
            raise DeploymentGenerationError(
                f"Deployment source {entry.destination!r} changed after planning "
                f"({'; '.join(mismatches)}); refusing to publish a mixed "
                "generation. Start a new deployment to pick up the current "
                "working tree."
            )
        os.chmod(target, entry.mode)
    except DeploymentGenerationError:
        target.unlink(missing_ok=True)
        raise
    except OSError as error:
        target.unlink(missing_ok=True)
        raise DeploymentPlanError(
            f"Cannot materialize deployment entry {entry.destination!r}: {error}"
        ) from error


def verify_plan_sources(plan: DeploymentPlan) -> None:
    """Re-scan the working tree and require it to match the frozen generation.

    Compares the current policy bytes, selected destination set, file types,
    modes, and SHA-256 content digests against ``plan``. Any replacement,
    deletion, addition, mode change, or type swap selected by the frozen
    policy raises :class:`DeploymentGenerationError`; an unchanged tree
    returns silently. Callers materialize only after this gate passes.
    """
    reasons: list[str] = []

    try:
        policy_stat = (plan.root / POLICY_FILENAME).lstat()
    except OSError:
        policy_stat = None
    if policy_stat is None or not stat.S_ISREG(policy_stat.st_mode):
        reasons.append(
            f"{POLICY_FILENAME} is no longer a regular file (deployment policy removed "
            "or replaced)"
        )
    else:
        try:
            policy_bytes = (plan.root / POLICY_FILENAME).read_bytes()
        except OSError as error:
            raise DeploymentPlanError(
                f"Cannot re-read deployment policy {plan.root / POLICY_FILENAME}: {error}"
            ) from error
        if _digest_bytes(policy_bytes) != plan.policy_digest:
            reasons.append(f"{POLICY_FILENAME} changed after planning")

    fresh = _scan_generation_fingerprints(plan)
    expected = {entry.destination: entry for entry in plan.entries}
    fresh_destinations = set(fresh)

    for entry in plan.entries:
        if entry.destination == POLICY_FILENAME:
            # The policy is compared directly above.
            continue
        current = fresh.get(entry.destination)
        if current is None:
            present = _locate_destination(plan.root, entry.destination)
            if present is None:
                reasons.append(f"{entry.destination!r} was deleted after planning")
            elif present == "directory":
                reasons.append(
                    f"{entry.destination!r} changed from a regular file to a directory"
                )
            else:
                reasons.append(
                    f"{entry.destination!r} is not a regular file anymore "
                    f"(replaced by a {present})"
                )
            continue
        if current.kind != "regular":
            reasons.append(
                f"{entry.destination!r} is not a regular file anymore "
                f"(replaced by a {current.kind})"
            )
            continue
        if current.digest != entry.digest:
            reasons.append(f"{entry.destination!r} content changed after planning")
        elif current.size != entry.size:
            reasons.append(f"{entry.destination!r} changed size after planning")
        if current.mode != entry.mode:
            reasons.append(f"{entry.destination!r} changed permissions after planning")

    for added in sorted(fresh_destinations - set(expected)):
        node = fresh[added]
        if node.kind == "regular":
            reasons.append(f"{added!r} was added after planning")
        else:
            reasons.append(
                f"{added!r} ({node.kind}) appeared after planning and is selected by "
                "the frozen policy"
            )

    if reasons:
        raise DeploymentGenerationError(_generation_error_message(plan, reasons))


def assembly_manifest_path(staging_directory: str | os.PathLike[str]) -> Path:
    """Return the isolated manifest path next to one materialized assembly."""
    staging = Path(os.path.abspath(os.fspath(staging_directory)))
    return staging.parent / ASSEMBLY_MANIFEST_NAME


def build_assembly_manifest(
    staging_directory: str | os.PathLike[str],
    *,
    generation: str,
) -> AssemblyManifest:
    """Fingerprint one complete, isolated assembly tree."""
    records = scan_assembly_tree(staging_directory)
    return AssemblyManifest(
        generation=generation,
        files=records,
        root_digest=assembly_root_digest(generation, records),
    )


def write_assembly_manifest(
    manifest: AssemblyManifest,
    path: str | os.PathLike[str],
) -> None:
    """Persist an assembly manifest as UTF-8 JSON."""
    target = Path(os.fspath(path))
    try:
        target.write_text(manifest.to_json() + "\n", encoding="utf-8")
    except OSError as error:
        raise DeploymentPlanError(
            f"Cannot write assembly manifest {target}: {error}"
        ) from error


def read_assembly_manifest(path: str | os.PathLike[str]) -> AssemblyManifest:
    """Load an assembly manifest previously written next to an assembly."""
    source = Path(os.fspath(path))
    try:
        return AssemblyManifest.from_json(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise DeploymentPlanError(
            f"Cannot read assembly manifest {source}: {error}"
        ) from error


def scan_assembly_tree(
    staging_directory: str | os.PathLike[str],
) -> tuple[AssemblyFileRecord, ...]:
    """Return sorted regular-file fingerprints for one isolated assembly."""
    root = Path(os.path.abspath(os.fspath(staging_directory)))
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise DeploymentPlanError(
            f"Cannot inspect assembly directory {root}: {error}"
        ) from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise DeploymentPlanError(f"Assembly directory is not a directory: {root}.")

    records: list[AssemblyFileRecord] = []

    def walk(directory: Path, parts: tuple[str, ...]) -> None:
        try:
            scanner = os.scandir(directory)
        except OSError as error:
            raise DeploymentPlanError(
                f"Cannot scan assembly directory {directory}: {error}"
            ) from error
        with scanner:
            for child in scanner:
                source = directory / child.name
                name = _normalize_source_component(child.name, source)
                destination_parts = (*parts, name)
                try:
                    child_stat = child.stat(follow_symlinks=False)
                except OSError as error:
                    raise DeploymentPlanError(
                        f"Cannot inspect assembly file {source}: {error}"
                    ) from error
                if stat.S_ISDIR(child_stat.st_mode):
                    walk(source, destination_parts)
                    continue
                if not stat.S_ISREG(child_stat.st_mode):
                    raise DeploymentPlanError(
                        f"Isolated assembly contains a non-regular file: {source}. "
                        "Only regular files can be frozen for deployment."
                    )
                digest, size = _hash_regular_file(source)
                if size != child_stat.st_size:
                    raise DeploymentGenerationError(
                        f"Assembly file {source} changed while being frozen "
                        f"({child_stat.st_size} bytes staged, {size} read)."
                    )
                records.append(
                    AssemblyFileRecord(
                        destination="/".join(destination_parts),
                        digest=digest,
                        size=child_stat.st_size,
                        mode=stat.S_IMODE(child_stat.st_mode),
                    )
                )

    walk(root, ())
    return tuple(sorted(records, key=lambda record: record.destination))


def assembly_root_digest(
    generation: str,
    records: Iterable[AssemblyFileRecord],
) -> str:
    """Hash the full file set of one assembly into one root identity."""
    hasher = hashlib.sha256()
    hasher.update(b"dallinger-assembly-1\n")
    hasher.update(generation.encode("utf-8"))
    hasher.update(b"\n")
    for record in sorted(records, key=lambda item: item.destination):
        hasher.update(record.destination.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(record.digest.encode("ascii"))
        hasher.update(b"\0")
        hasher.update(str(record.size).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(f"{record.mode:04o}".encode("ascii"))
        hasher.update(b"\n")
    return _digest_from_hash(hasher)


def verify_assembly_directory(
    staging_directory: str | os.PathLike[str],
    manifest: AssemblyManifest,
    *,
    allow: Iterable[str] = (),
) -> None:
    """Require an isolated tree to match its frozen assembly manifest.

    ``allow`` names root-relative destinations the framework legitimately
    rewrites or adds after freezing (for example the generated
    ``Dockerfile``). They are exempt from comparison in both directions;
    their final bytes are still the framework's own deterministic output.
    Every other destination must match the frozen manifest exactly.
    """
    actual = scan_assembly_tree(staging_directory)
    _compare_assembly_records(
        manifest,
        actual,
        allowed=frozenset(allow),
        context="Docker context",
    )


def verify_zip_archive(
    archive_path: str | os.PathLike[str],
    manifest: AssemblyManifest,
    *,
    allow: Iterable[str] = (),
) -> None:
    """Require a zip archive's members and contents to match the manifest."""
    archive = Path(os.fspath(archive_path))
    allowed = frozenset(allow)
    records: list[AssemblyFileRecord] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [
                info
                for info in bundle.infolist()
                if not info.is_dir() and not info.filename.endswith("/")
            ]
            destinations = [_zip_member_destination(info.filename) for info in members]
            prefix = _common_root_prefix(destinations)
            for info, destination in zip(members, destinations, strict=True):
                relative = destination[len(prefix) :] if prefix else destination
                with bundle.open(info, "r") as member:
                    digest, size = _hash_stream(member)
                records.append(
                    AssemblyFileRecord(
                        destination=relative,
                        digest=digest,
                        size=size,
                        mode=_zip_entry_mode(info),
                    )
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise DeploymentPlanError(
            f"Cannot read deployment archive {archive}: {error}"
        ) from error
    _compare_assembly_records(
        manifest,
        tuple(sorted(records, key=lambda record: record.destination)),
        allowed=allowed,
        context="Deployment archive",
        compare_modes=False,
    )


def _compare_assembly_records(
    manifest: AssemblyManifest,
    actual: Iterable[AssemblyFileRecord],
    *,
    allowed: frozenset[str],
    context: str,
    compare_modes: bool = True,
) -> None:
    expected = {record.destination: record for record in manifest.files}
    actual_map = {record.destination: record for record in actual}
    reasons: list[str] = []

    # Allowed destinations are framework-owned rewrites/additions performed
    # after the freeze (for example the generated Dockerfile), so they are
    # excluded from comparison in both directions; everything else must
    # match byte for byte.

    for destination in sorted(set(actual_map) - set(expected)):
        if destination in allowed:
            continue
        reasons.append(f"{destination!r} is present but absent from the frozen manifest")

    for destination, record in sorted(expected.items()):
        if destination in allowed:
            continue
        current = actual_map.get(destination)
        if current is None:
            reasons.append(f"{destination!r} is missing")
            continue
        if current.digest != record.digest:
            reasons.append(f"{destination!r} content does not match the frozen manifest")
        elif current.size != record.size:
            reasons.append(f"{destination!r} size does not match the frozen manifest")
        if compare_modes and current.mode != record.mode:
            reasons.append(
                f"{destination!r} permissions do not match the frozen manifest"
            )

    if reasons:
        displayed = "; ".join(reasons[:10])
        if len(reasons) > 10:
            displayed += f"; … {len(reasons) - 10} more"
        raise DeploymentGenerationError(
            f"{context} drifts from frozen deployment generation "
            f"{manifest.generation[:12]}: {displayed}. Refusing to publish this "
            "generation; rebuild from its isolated artifact or start a new "
            "deployment."
        )


def _finalize_directory_candidate(
    candidate: _TraversedDirectoryCandidate,
    ordered_destinations: tuple[str, ...],
) -> DeploymentDirectoryLinkCandidate:
    """Map one traversal candidate to its contiguous ordered plan-entry span."""
    prefix = candidate.destination + "/"
    entry_start = bisect.bisect_left(ordered_destinations, prefix)
    entry_stop = bisect.bisect_left(
        ordered_destinations, candidate.destination + "0", lo=entry_start
    )
    if entry_stop - entry_start != candidate.entry_count:
        raise DeploymentPlanError(
            "Deployment directory candidate does not cover exactly its planned "
            f"descendants: {candidate.source}."
        )
    return DeploymentDirectoryLinkCandidate(
        source=candidate.source,
        destination=candidate.destination,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )


def _require_string_list(value: object, key: str) -> list[str]:
    """Require a TOML array of strings for one policy key."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DeploymentPolicyError(
            f"Deployment policy {key} must be an array of strings."
        )
    return value


def _optional_string_list(raw_policy: dict, key: str) -> list[str]:
    """Return one optional TOML string array, or an empty list if omitted."""
    if key not in raw_policy:
        return []
    return _require_string_list(raw_policy[key], key)


def _normalized_unique_strings(
    values: list[str],
    *,
    validator: Callable[[str], str],
    duplicate_label: str,
) -> list[str]:
    """Validate strings, reject normalized duplicates, and return them in order."""
    normalized_values: list[str] = []
    exact_entries: set[str] = set()
    for raw_value in values:
        normalized = validator(raw_value)
        if normalized in exact_entries:
            raise DeploymentPolicyError(
                f"Duplicate normalized {duplicate_label}: {normalized!r}."
            )
        exact_entries.add(normalized)
        normalized_values.append(normalized)
    return normalized_values


def _reject_unsupported_globs(
    value: str, *, where: str, reported: str | None = None
) -> None:
    """Reject Git-style glob characters in one policy field."""
    if any(character in _GLOB_CHARACTERS for character in value):
        raise DeploymentPolicyError(
            f"Glob syntax is unsupported in {where}: {(reported or value)!r}."
        )


def _validate_policy_path(value: str) -> str:
    """Validate and NFC-normalize one literal root-relative exclusion."""
    if not value:
        raise DeploymentPolicyError("Exclusion paths must not be empty.")
    if "\\" in value:
        raise DeploymentPolicyError(
            f"Exclusion paths must use POSIX separators: {value!r}."
        )
    if value.startswith("/"):
        raise DeploymentPolicyError(
            f"Exclusion paths must be relative to the experiment root: {value!r}."
        )
    if value.startswith("!"):
        raise DeploymentPolicyError(
            f"Negated exclusion paths are unsupported: {value!r}."
        )
    _reject_unsupported_globs(value, where="exclude.paths")

    components = value.split("/")
    if any(not component for component in components):
        raise DeploymentPolicyError(
            f"Exclusion paths must not contain empty components: {value!r}."
        )

    normalized_parts = tuple(
        _normalize_path_component(
            component,
            context=f"exclusion path {value!r}",
            error_type=DeploymentPolicyError,
        )
        for component in components
    )
    normalized = "/".join(normalized_parts)
    _reject_reserved_exclusion(normalized_parts)
    return normalized


def _validate_exclude_name(value: str) -> str:
    """Validate one basename omitted in every directory."""
    if not value:
        raise DeploymentPolicyError("exclude.names entries must not be empty.")
    if "/" in value or "\\" in value:
        raise DeploymentPolicyError(
            f"exclude.names entries must be a single path component: {value!r}."
        )
    if value.startswith("!"):
        raise DeploymentPolicyError(
            f"Negated exclude.names entries are unsupported: {value!r}."
        )
    _reject_unsupported_globs(value, where="exclude.names")
    normalized = _normalize_path_component(
        value,
        context=f"exclude.names entry {value!r}",
        error_type=DeploymentPolicyError,
    )
    _reject_reserved_exclusion((normalized,))
    return normalized


def _validate_exclude_suffix(value: str) -> str:
    """Validate one ``.db``-style suffix omitted in every directory."""
    if "/" in value or "\\" in value:
        raise DeploymentPolicyError(
            f"exclude.suffixes entries must be a single path component: {value!r}."
        )
    _reject_unsupported_globs(value, where="exclude.suffixes")
    normalized = unicodedata.normalize("NFC", value)
    if (
        not normalized.startswith(".")
        or normalized in {".", ".."}
        or len(normalized) < 2
    ):
        raise DeploymentPolicyError(
            f"exclude.suffixes entries must look like '.db': {value!r}."
        )
    _reject_reserved_exclusion((normalized,))
    return normalized


def _normalize_source_component(name: str, source: Path) -> str:
    """Normalize one filesystem name for use as a POSIX destination component."""
    return _normalize_path_component(
        name,
        context=f"deployment source {os.fspath(source)!r}",
        error_type=DeploymentPlanError,
    )


def _normalize_path_component(
    component: str,
    *,
    context: str,
    error_type: type[_ErrorType],
) -> str:
    """Validate and NFC-normalize one destination component."""
    normalized = unicodedata.normalize("NFC", component)
    if not normalized or normalized in {".", ".."}:
        raise error_type(f"Invalid path component in {context}: {component!r}.")
    return normalized


def _is_excluded(
    destination_parts: tuple[str, ...], exclusions: AbstractSet[str]
) -> bool:
    """Return whether any ancestor destination is excluded."""
    return any(
        "/".join(destination_parts[:depth]) in exclusions
        for depth in range(1, len(destination_parts) + 1)
    )


def _has_exclusion_at_or_below(
    destination_parts: tuple[str, ...], exclusions: AbstractSet[str]
) -> bool:
    """Return whether a literal exclusion is this directory or a descendant."""
    if not exclusions:
        return False
    if not destination_parts:
        return True
    destination = "/".join(destination_parts)
    if destination in exclusions:
        return True
    prefix = destination + "/"
    return any(item.startswith(prefix) for item in exclusions)


def _reject_reserved_exclusion(destination_parts: tuple[str, ...]) -> None:
    """Reject exclusions that target required or reserved destinations."""
    if _reserved_kind(destination_parts) is None:
        return
    raise DeploymentPolicyError(
        "Deployment policy cannot exclude required or reserved path "
        f"{'/'.join(destination_parts)!r}."
    )


def _is_omitted_anywhere(
    name: str,
    anywhere_names: AbstractSet[str],
    anywhere_suffixes: tuple[str, ...],
) -> bool:
    """Return whether a basename is omitted in every directory."""
    return name in anywhere_names or bool(
        anywhere_suffixes and name.endswith(anywhere_suffixes)
    )


def _reserved_kind(destination_parts: tuple[str, ...]) -> str | None:
    """Classify paths at or beneath non-overridable prefixes."""
    root_name = destination_parts[0]
    if root_name == POLICY_FILENAME:
        return "policy"
    if any(part in _VCS_METADATA_NAMES for part in destination_parts):
        return "vcs"
    if root_name == "config.txt":
        return "configuration"
    if root_name in _GENERATED_ROOT_DESTINATION_NAMES:
        return "generated"
    if any(
        part == ".slugignore" or part.endswith(".dockerignore")
        for part in destination_parts
    ):
        return "backend-ignore"
    return None


def _require_posix_support(error_type: type[_ErrorType]) -> None:
    """Fail closed on non-POSIX platforms for this prototype."""
    if os.name != "posix":
        raise error_type("Deployment planning currently requires a POSIX filesystem.")


def _assert_source_under_root(root: Path, source: Path) -> None:
    """Require a planned source path to remain inside its plan root."""
    try:
        relative_source = source.relative_to(root)
    except ValueError as error:
        raise DeploymentPlanError(
            f"Deployment source is outside its plan root: {source}."
        ) from error
    if not relative_source.parts or ".." in relative_source.parts:
        raise DeploymentPlanError(
            f"Deployment source is not a safe root-relative path: {source}."
        )


def _read_policy_snapshot(source: Path) -> _PolicySnapshot:
    """Parse a policy file without following a final-component symlink."""
    try:
        source_stat = source.lstat()
    except OSError as error:
        raise DeploymentPolicyError(
            f"Cannot inspect deployment policy {source}: {error}"
        ) from error

    if stat.S_ISLNK(source_stat.st_mode):
        raise DeploymentPolicyError(
            f"Deployment policy {source} must not be a symbolic link."
        )
    if not stat.S_ISREG(source_stat.st_mode):
        raise DeploymentPolicyError(
            f"Deployment policy {source} must be a regular file."
        )

    try:
        content = source.read_bytes()
    except OSError as error:
        raise DeploymentPolicyError(
            f"Cannot read deployment policy {source}: {error}"
        ) from error

    try:
        policy_text = content.decode("utf-8")
        raw_policy = tomllib.loads(policy_text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise DeploymentPolicyError(
            f"Invalid TOML in deployment policy {source}: {error}"
        ) from error
    return _PolicySnapshot(
        policy=_parse_policy(raw_policy),
        size=source_stat.st_size,
        mode=stat.S_IMODE(source_stat.st_mode),
        digest=_digest_bytes(content),
    )


def _source_type(mode: int) -> str:
    if stat.S_ISFIFO(mode):
        return "FIFO"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character-device"
    if stat.S_ISBLK(mode):
        return "block-device"
    return "special-file"


def _digest_from_hash(hasher: "hashlib._Hash") -> str:
    """Render a finalized content hash as a prefixed digest."""
    return DIGEST_PREFIX + hasher.hexdigest()


def _digest_bytes(content: bytes) -> str:
    """Return the prefixed SHA-256 digest of one in-memory blob."""
    return _digest_from_hash(hashlib.sha256(content))


def _hash_stream(stream) -> tuple[str, int]:
    """Hash one binary stream, returning its prefixed digest and byte count."""
    hasher = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(HASH_CHUNK_SIZE)
        if not chunk:
            break
        hasher.update(chunk)
        size += len(chunk)
    return _digest_from_hash(hasher), size


def _hash_regular_file(source: Path) -> tuple[str, int]:
    """Hash one regular file without following a final-component symlink."""
    with open(source, "rb") as stream:
        return _hash_stream(stream)


def _plan_generation(
    policy_digest: str,
    entries: tuple[DeploymentPlanEntry, ...],
) -> str:
    """Compute the content generation identity for one frozen plan."""
    hasher = hashlib.sha256()
    hasher.update(b"dallinger-deployment-generation-1\n")
    hasher.update(policy_digest.encode("ascii"))
    hasher.update(b"\n")
    for entry in entries:
        hasher.update(entry.destination.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(entry.digest.encode("ascii"))
        hasher.update(b"\0")
        hasher.update(str(entry.size).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(f"{entry.mode:04o}".encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


@dataclass(frozen=True)
class _FreshNode:
    kind: str
    digest: str | None = None
    size: int | None = None
    mode: int | None = None


def _scan_generation_fingerprints(plan: DeploymentPlan) -> dict[str, _FreshNode]:
    """Re-scan selected sources under the frozen policy.

    Mirrors the selection rules of :func:`build_deployment_plan` but never
    raises on a bad node: the node is fingerprinted or classified so the
    caller can report it as generation drift.
    """
    root = plan.root
    policy = plan.policy
    exclusions = frozenset(policy.exclude_paths)
    anywhere_names = frozenset(policy.exclude_names)
    anywhere_suffixes = policy.exclude_suffixes
    nodes: dict[str, _FreshNode] = {}

    def walk(directory: Path, destination_parts: tuple[str, ...]) -> None:
        try:
            scanner = os.scandir(directory)
        except OSError as error:
            raise DeploymentPlanError(
                f"Cannot re-scan experiment directory {directory}: {error}"
            ) from error
        with scanner:
            for child in scanner:
                source = directory / child.name
                name = _normalize_source_component(child.name, source)
                child_parts = (*destination_parts, name)
                destination = "/".join(child_parts)

                if _is_excluded(child_parts, exclusions):
                    continue
                if _is_omitted_anywhere(name, anywhere_names, anywhere_suffixes):
                    continue
                reserved_kind = _reserved_kind(child_parts)
                if reserved_kind == "vcs":
                    if destination_parts:
                        nodes[destination] = _FreshNode("repository metadata")
                    continue
                if reserved_kind is not None:
                    continue

                try:
                    source_stat = child.stat(follow_symlinks=False)
                except OSError as error:
                    raise DeploymentPlanError(
                        f"Cannot re-inspect deployment source {source}: {error}"
                    ) from error

                if stat.S_ISDIR(source_stat.st_mode):
                    walk(source, child_parts)
                elif stat.S_ISREG(source_stat.st_mode):
                    digest, size = _hash_regular_file(source)
                    nodes[destination] = _FreshNode(
                        "regular",
                        digest=digest,
                        size=size,
                        mode=stat.S_IMODE(source_stat.st_mode),
                    )
                elif stat.S_ISLNK(source_stat.st_mode):
                    nodes[destination] = _FreshNode("symbolic link")
                else:
                    nodes[destination] = _FreshNode(
                        _source_type(source_stat.st_mode)
                    )

    walk(root, ())
    return nodes


def _locate_destination(root: Path, destination: str) -> str | None:
    """Classify a planned destination that disappeared from a fresh scan."""
    target = root.joinpath(*destination.split("/"))
    try:
        target_stat = target.lstat()
    except OSError:
        return None
    if stat.S_ISDIR(target_stat.st_mode):
        return "directory"
    if stat.S_ISLNK(target_stat.st_mode):
        return "symbolic link"
    if stat.S_ISREG(target_stat.st_mode):
        return "unselected regular file"
    return _source_type(target_stat.st_mode)


def _generation_error_message(plan: DeploymentPlan, reasons: list[str]) -> str:
    """Render one fail-closed generation error listing every drift found."""
    displayed = "; ".join(reasons[:10])
    if len(reasons) > 10:
        displayed += f"; … {len(reasons) - 10} more"
    return (
        f"Working tree no longer matches frozen deployment generation "
        f"{plan.generation[:12]} ({len(reasons)} change(s)): {displayed}. "
        "The planned archive or image will not be published for this "
        "generation. Finish or discard the attempt, then start a new "
        "deployment to replan against the current tree."
    )


def _zip_member_destination(name: str) -> str:
    """Normalize one zip member name to a root-relative POSIX destination."""
    normalized = name.replace("\\", "/")
    while normalized.startswith("/"):
        normalized = normalized[1:]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def _common_root_prefix(destinations: list[str]) -> str:
    """Return the shared top-level directory prefix of a member set, if any."""
    tops = {destination.split("/", 1)[0] for destination in destinations if destination}
    if len(tops) != 1:
        return ""
    top = next(iter(tops))
    prefix = top + "/"
    if all(destination == top or destination.startswith(prefix) for destination in destinations):
        return prefix
    return ""


def _zip_entry_mode(info: zipfile.ZipInfo) -> int:
    """Best-effort POSIX mode recorded on one zip member."""
    return stat.S_IMODE((info.external_attr >> 16) & 0xFFFF)
