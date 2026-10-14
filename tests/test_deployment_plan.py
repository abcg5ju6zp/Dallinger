import json
import os
import re
import shutil
import unicodedata
import zipfile

import pytest

import dallinger.deployment_plan as deployment_plan
from dallinger.deployment_plan import (
    DeploymentGenerationError,
    DeploymentPlanError,
    DeploymentPolicyError,
    build_deployment_plan,
    parse_deployment_policy,
    validate_explicit_provider_destination,
)
from tests.helpers import write_deployment_policy, write_files

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="deployment planning currently requires a POSIX filesystem",
)


def test_parse_valid_policy_normalizes_and_sorts_exclusions(tmp_path):
    decomposed = unicodedata.normalize("NFD", "café")
    policy = parse_deployment_policy(
        write_deployment_policy(tmp_path, ["z/data", decomposed])
    )

    assert policy.version == 1
    assert policy.exclude_paths == ("café", "z/data")
    assert policy.exclude_names == ()
    assert policy.exclude_suffixes == ()


def test_parse_valid_policy_normalizes_and_sorts_names(tmp_path):
    decomposed = unicodedata.normalize("NFD", "café")
    policy = parse_deployment_policy(
        write_deployment_policy(tmp_path, names=["zcache", decomposed])
    )

    assert policy.exclude_names == ("café", "zcache")


@pytest.mark.parametrize(
    "contents, message",
    [
        ("exclude = []\n", "version"),
        ("version = 1\n", "exclude"),
        ('version = 1\n[exclude]\nextra = "x"\n', "Unknown exclude"),
        ("version = 2\n[exclude]\n", "version"),
        ("version = true\n[exclude]\n", "version"),
        ('version = 1\nexclude = "data"\n', "table"),
        ("version = 1\nexclude = []\n", "table"),
        ('version = 1\n[exclude]\npaths = "data"\n', "array of strings"),
        ("version = 1\n[exclude]\npaths = [1]\n", "array of strings"),
        ('version = 1\n[exclude]\nnames = "cache"\n', "array of strings"),
        ("version = 1\n[exclude]\nnames = [1]\n", "array of strings"),
        ('version = 1\n[exclude]\nsuffixes = ".db"\n', "array of strings"),
        ("version = 1\n[exclude]\nsuffixes = [1]\n", "array of strings"),
        (
            "version = 1\n[exclude]\nlegacy_diff_acknowledgement = "
            f'"sha256:{"a" * 64}"\n',
            "Unknown exclude",
        ),
        ("version = 1\nfoo = 1\n[exclude]\n", "Unknown deployment policy"),
    ],
)
def test_parse_rejects_unknown_missing_version_and_invalid_types(
    tmp_path, contents, message
):
    path = tmp_path / "deploy.toml"
    path.write_text(contents)

    with pytest.raises(DeploymentPolicyError, match=message):
        parse_deployment_policy(path)


@pytest.mark.parametrize(
    "excluded",
    [
        "",
        "/absolute",
        "has\\slash",
        "!negated",
        "glob*",
        "a//b",
        ".",
        "..",
        "deploy.toml",
        ".git",
        "config.txt",
        "constraints.txt",
        "requirements.txt",
        "runtime.txt",
        "experiment_id.txt",
        ".dockerignore",
    ],
)
def test_parse_rejects_nonliteral_or_unsafe_paths(tmp_path, excluded):
    with pytest.raises(DeploymentPolicyError):
        parse_deployment_policy(write_deployment_policy(tmp_path, [excluded]))


def test_parse_rejects_duplicate_normalized_paths(tmp_path):
    decomposed = unicodedata.normalize("NFD", "café")
    with pytest.raises(DeploymentPolicyError, match="Duplicate"):
        parse_deployment_policy(write_deployment_policy(tmp_path, ["café", decomposed]))


@pytest.mark.parametrize(
    "excluded",
    [
        "",
        "/absolute",
        "has/slash",
        "has\\slash",
        "!negated",
        "glob*",
        "*",
        "*.",
        "*db",
        "**",
        "*.*",
        ".",
        "..",
        "deploy.toml",
        ".git",
        "config.txt",
        ".dockerignore",
    ],
)
def test_parse_rejects_nonliteral_or_unsafe_exclude_names(tmp_path, excluded):
    with pytest.raises(DeploymentPolicyError):
        parse_deployment_policy(write_deployment_policy(tmp_path, names=[excluded]))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "exclude.names"),
        ("!secret", "exclude.names"),
        ("config.txt", "reserved"),
    ],
)
def test_parse_exclude_name_errors_name_the_policy_key(tmp_path, value, message):
    with pytest.raises(DeploymentPolicyError, match=message):
        parse_deployment_policy(write_deployment_policy(tmp_path, names=[value]))


def test_parse_accepts_exclude_suffixes(tmp_path):
    policy = parse_deployment_policy(
        write_deployment_policy(tmp_path, suffixes=[".dmg", ".db"])
    )

    assert policy.exclude_suffixes == (".db", ".dmg")


@pytest.mark.parametrize(
    "suffix",
    ["", ".", "..", "db", "*.db", ".d*b", ".db*", "/.db", "a.db"],
)
def test_parse_rejects_nonliteral_or_unsafe_exclude_suffixes(tmp_path, suffix):
    with pytest.raises(DeploymentPolicyError):
        parse_deployment_policy(write_deployment_policy(tmp_path, suffixes=[suffix]))


@pytest.mark.parametrize(
    ("value", "field", "where"),
    [
        ("glob*", "paths", "exclude.paths"),
        ("*.db", "paths", "exclude.paths"),
        ("glob*", "names", "exclude.names"),
        ("*.db", "names", "exclude.names"),
        ("*.db", "suffixes", "exclude.suffixes"),
        (".d*b", "suffixes", "exclude.suffixes"),
    ],
)
def test_parse_glob_errors_name_the_policy_key(tmp_path, value, field, where):
    with pytest.raises(
        DeploymentPolicyError,
        match=rf"Glob syntax is unsupported in {where}: {re.escape(repr(value))}",
    ):
        parse_deployment_policy(write_deployment_policy(tmp_path, **{field: [value]}))


def test_parse_rejects_duplicate_normalized_exclude_names(tmp_path):
    decomposed = unicodedata.normalize("NFD", "café")
    with pytest.raises(DeploymentPolicyError, match="Duplicate"):
        parse_deployment_policy(
            write_deployment_policy(tmp_path, names=["café", decomposed])
        )


def test_explicit_provider_rejects_reserved_destination_prefixes():
    with pytest.raises(DeploymentPlanError, match="reserved"):
        validate_explicit_provider_destination("config.txt")
    with pytest.raises(DeploymentPlanError, match="reserved"):
        validate_explicit_provider_destination(".dockerignore")


def test_explicit_provider_allows_nested_config_and_normalizes_nfc():
    decomposed = unicodedata.normalize("NFD", "café")
    assert (
        validate_explicit_provider_destination(f"nested/{decomposed}/file.txt")
        == "nested/café/file.txt"
    )


def test_parse_rejects_symlinked_policy(tmp_path):
    target = tmp_path / "policy.toml"
    target.write_text("version = 1\n[exclude]\n")
    (tmp_path / "deploy.toml").symlink_to(target)

    with pytest.raises(DeploymentPolicyError, match="symbolic link"):
        parse_deployment_policy(tmp_path / "deploy.toml")


def test_plan_is_sorted_and_independent_of_creation_order(tmp_path):
    write_deployment_policy(tmp_path)
    write_files(tmp_path, {"z.txt": "z", "a.txt": "a", "m/n.txt": "n"})

    plan = build_deployment_plan(tmp_path)

    assert [entry.destination for entry in plan.entries] == [
        "a.txt",
        "deploy.toml",
        "m/n.txt",
        "z.txt",
    ]


def test_plan_root_exclude_does_not_omit_nested_same_name(tmp_path):
    write_deployment_policy(tmp_path, ["__pycache__", ".env"])
    write_files(
        tmp_path,
        {
            "__pycache__/root.pyc": "root cache",
            "pkg/__pycache__/x.pyc": "nested cache",
            ".env": "root secret",
            "nested/.env": "nested secret",
        },
    )

    plan = build_deployment_plan(tmp_path)

    assert plan.destinations == {
        "deploy.toml",
        "nested/.env",
        "pkg/__pycache__/x.pyc",
    }


def test_plan_exclude_names_and_suffixes_omit_nested_junk(tmp_path):
    write_deployment_policy(
        tmp_path,
        paths=["data"],
        names=["__pycache__", ".env"],
        suffixes=[".db"],
    )
    write_files(
        tmp_path,
        {
            "pkg/__pycache__/x.pyc": "nested cache",
            "pkg/mod.py": "code",
            "nested/.env": "nested secret",
            "data/local.csv": "export",
            "static/data/tone.wav": "stimulus",
            "illegit.db": "root db",
            "legit_dir/illegit.db": "nested db",
            "keep.txt": "ok",
        },
    )

    plan = build_deployment_plan(tmp_path)

    assert plan.destinations == {
        "deploy.toml",
        "keep.txt",
        "pkg/mod.py",
        "static/data/tone.wav",
    }


def test_starter_policy_omits_nested_junk_and_keeps_nested_data(tmp_path):
    from dallinger.command_line.deployment_files import (
        _STARTER_EXCLUDE_NAMES,
        _STARTER_EXCLUDE_PATHS,
        _STARTER_EXCLUDE_SUFFIXES,
    )

    write_deployment_policy(
        tmp_path,
        paths=_STARTER_EXCLUDE_PATHS,
        names=_STARTER_EXCLUDE_NAMES,
        suffixes=_STARTER_EXCLUDE_SUFFIXES,
    )
    write_files(
        tmp_path,
        {
            "pkg/__pycache__/x.pyc": "nested cache",
            "pkg/mod.py": "code",
            "nested/.env": "nested secret",
            "illegit.db": "root db",
            "legit_dir/illegit.db": "nested db",
            "static/data/tone.wav": "stimulus",
        },
    )

    plan = build_deployment_plan(tmp_path)

    assert "pkg/__pycache__/x.pyc" not in plan
    assert "nested/.env" not in plan
    assert "illegit.db" not in plan
    assert "legit_dir/illegit.db" not in plan
    assert "pkg/mod.py" in plan
    assert "static/data/tone.wav" in plan


def test_plan_empty_exclude_still_includes_nested_pycache(tmp_path):
    write_deployment_policy(tmp_path)
    write_files(
        tmp_path,
        {
            "pkg/__pycache__/x.pyc": "nested cache",
            "nested/.env": "nested secret",
            "illegit.db": "root db",
        },
    )

    plan = build_deployment_plan(tmp_path)

    assert "pkg/__pycache__/x.pyc" in plan
    assert "nested/.env" in plan
    assert "illegit.db" in plan


def test_plan_exclude_suffix_prunes_matching_directory(tmp_path):
    write_deployment_policy(tmp_path, suffixes=[".db"])
    write_files(
        tmp_path,
        {
            "complete/keep.txt": "ok",
            "store.db/secret.txt": "hidden",
            "store.db.bak": "kept",
        },
    )

    plan = build_deployment_plan(tmp_path)
    destinations = {
        candidate.destination for candidate in plan.directory_link_candidates
    }

    assert plan.destinations == {
        "complete/keep.txt",
        "deploy.toml",
        "store.db.bak",
    }
    assert "complete" in destinations
    assert "store.db" not in destinations


def test_plan_applies_literal_prefixes_allows_missing_and_includes_policy(tmp_path):
    write_deployment_policy(tmp_path, ["static/private", "missing"])
    write_files(
        tmp_path,
        {
            "static/private/secret.txt": "excluded descendant",
            "static/public/asset.txt": "included",
        },
    )

    plan = build_deployment_plan(tmp_path)

    assert plan.destinations == {"deploy.toml", "static/public/asset.txt"}
    assert "deploy.toml" in plan
    assert "static/private/secret.txt" not in plan
    assert plan.total_size == sum(entry.size for entry in plan.entries)


def test_plan_omits_reserved_source_paths(tmp_path):
    write_deployment_policy(tmp_path)
    write_files(
        tmp_path,
        {
            "config.txt": "raw secret",
            "constraints.txt": "pins",
            "requirements.txt": "dallinger",
            "runtime.txt": "python-3.13",
            "experiment_id.txt": "old-id",
            ".git/objects/example": "metadata",
            ".dockerignore": "*",
            "Dockerfile.production.dockerignore": "*",
            ".slugignore": "*",
            "nested/config.txt": "ordinary nested file",
            ".gitignore": "*.secret",
        },
    )

    plan = build_deployment_plan(tmp_path)

    assert plan.destinations == {
        ".gitignore",
        "deploy.toml",
        "nested/config.txt",
    }


def test_plan_records_entry_metadata_and_membership(tmp_path):
    write_deployment_policy(tmp_path)
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)

    plan = build_deployment_plan(tmp_path)
    entry = next(item for item in plan.entries if item.destination == "run.sh")

    assert entry.source == script
    assert entry.size == script.stat().st_size
    assert entry.executable is True
    assert entry.mode & 0o111
    assert "run.sh" in plan


def test_plan_records_only_fully_selected_directory_link_candidates(tmp_path):
    write_deployment_policy(tmp_path, ["mixed/private"])
    write_files(
        tmp_path,
        {
            "complete/first.txt": "first",
            "complete/nested/second.txt": "second",
            "mixed/private/secret.txt": "secret",
            "mixed/public/visible.txt": "visible",
        },
    )

    plan = build_deployment_plan(tmp_path)
    candidates = {
        candidate.destination: candidate for candidate in plan.directory_link_candidates
    }

    assert "complete" in candidates
    assert "complete/nested" in candidates
    assert "mixed" not in candidates
    assert "mixed/public" in candidates
    complete = candidates["complete"]
    assert tuple(
        entry.destination
        for entry in plan.entries[complete.entry_start : complete.entry_stop]
    ) == ("complete/first.txt", "complete/nested/second.txt")


def test_plan_rejects_selected_symlink_but_prunes_excluded_symlink(tmp_path):
    write_deployment_policy(tmp_path, ["excluded-link"])
    target = tmp_path / "target.txt"
    target.write_text("target")
    (tmp_path / "excluded-link").symlink_to(target)
    selected = tmp_path / "selected-link"
    selected.symlink_to(target)

    with pytest.raises(DeploymentPlanError, match="Symbolic links"):
        build_deployment_plan(tmp_path)

    selected.unlink()
    assert "excluded-link" not in build_deployment_plan(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_plan_rejects_special_files(tmp_path):
    write_deployment_policy(tmp_path)
    os.mkfifo(tmp_path / "events.fifo")

    with pytest.raises(DeploymentPlanError, match="FIFO"):
        build_deployment_plan(tmp_path)


def test_plan_rejects_nested_repositories(tmp_path):
    write_deployment_policy(tmp_path)
    marker = tmp_path / "vendor" / ".git"
    marker.parent.mkdir()
    marker.mkdir()

    with pytest.raises(DeploymentPlanError, match="Nested repository"):
        build_deployment_plan(tmp_path)


def test_plan_fails_closed_on_non_posix_platforms(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment_plan.os, "name", "nt")

    with pytest.raises(DeploymentPlanError, match="POSIX"):
        build_deployment_plan(tmp_path)


def test_exclusion_lookup_checks_ancestors():
    exclusions = frozenset({"static/private"})
    assert deployment_plan._is_excluded(("static", "private", "x"), exclusions)
    assert not deployment_plan._is_excluded(("static", "public", "x"), exclusions)
    assert deployment_plan._has_exclusion_at_or_below(("static",), exclusions)
    assert not deployment_plan._has_exclusion_at_or_below(("other",), exclusions)


def _entry(plan, destination):
    return next(item for item in plan.entries if item.destination == destination)


def test_plan_entries_record_sha256_content_digests(tmp_path):
    import hashlib

    write_deployment_policy(tmp_path)
    write_files(tmp_path, {"a.txt": "abc"})

    plan = build_deployment_plan(tmp_path)
    entry = _entry(plan, "a.txt")

    assert entry.digest == "sha256:" + hashlib.sha256(b"abc").hexdigest()
    assert plan.policy_digest == _entry(plan, "deploy.toml").digest
    assert re.fullmatch(r"[0-9a-f]{64}", plan.generation)


def test_plan_generation_is_identical_for_identical_trees(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        write_deployment_policy(root)
        write_files(root, {"a.txt": "a", "m/n.txt": "n", "run.sh": "#!/bin/sh\n"})
        (root / "run.sh").chmod(0o755)

    assert build_deployment_plan(first).generation == (
        build_deployment_plan(second).generation
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda root: (root / "a.txt").write_text("changed"), id="content"),
        pytest.param(lambda root: (root / "run.sh").chmod(0o700), id="mode"),
        pytest.param(lambda root: write_files(root, {"new.txt": "new"}), id="added"),
        pytest.param(lambda root: (root / "a.txt").unlink(), id="deleted"),
    ],
)
def test_plan_generation_changes_with_any_input_drift(tmp_path, mutate):
    write_deployment_policy(tmp_path)
    write_files(tmp_path, {"a.txt": "a"})
    run_script = tmp_path / "run.sh"
    run_script.write_text("#!/bin/sh\n")
    run_script.chmod(0o755)
    before = build_deployment_plan(tmp_path).generation

    mutate(tmp_path)

    assert build_deployment_plan(tmp_path).generation != before


def test_plan_generation_changes_when_policy_changes(tmp_path):
    write_deployment_policy(tmp_path)
    write_files(tmp_path, {"a.txt": "a"})
    before = build_deployment_plan(tmp_path).generation

    write_deployment_policy(tmp_path, ["a.txt"])

    assert build_deployment_plan(tmp_path).generation != before


def test_verify_plan_sources_accepts_unchanged_tree(tmp_path):
    write_deployment_policy(tmp_path)
    write_files(tmp_path, {"a.txt": "a", "d/b.txt": "b"})
    plan = build_deployment_plan(tmp_path)

    deployment_plan.verify_plan_sources(plan)


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        pytest.param(
            lambda root: (root / "a.txt").write_text("changed"),
            "content changed",
            id="content",
        ),
        pytest.param(
            lambda root: (root / "a.txt").unlink(),
            "deleted",
            id="deleted",
        ),
        pytest.param(
            lambda root: write_files(root, {"new.txt": "new"}),
            "added",
            id="added",
        ),
        pytest.param(
            lambda root: (root / "run.sh").chmod(0o700),
            "permissions",
            id="mode",
        ),
    ],
)
def test_verify_plan_sources_rejects_drift(tmp_path, mutate, fragment):
    write_deployment_policy(tmp_path)
    write_files(tmp_path, {"a.txt": "a"})
    run_script = tmp_path / "run.sh"
    run_script.write_text("#!/bin/sh\n")
    run_script.chmod(0o755)
    plan = build_deployment_plan(tmp_path)

    mutate(tmp_path)

    with pytest.raises(DeploymentGenerationError, match=fragment):
        deployment_plan.verify_plan_sources(plan)


def test_verify_plan_sources_rejects_file_to_directory_swap(tmp_path):
    write_deployment_policy(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("a")
    plan = build_deployment_plan(tmp_path)

    target.unlink()
    (target / "child.txt").parent.mkdir()
    target.joinpath("child.txt").write_text("inside")

    with pytest.raises(DeploymentGenerationError, match="directory"):
        deployment_plan.verify_plan_sources(plan)


def test_verify_plan_sources_rejects_symlink_replacement(tmp_path):
    write_deployment_policy(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("a")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    plan = build_deployment_plan(tmp_path)

    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(DeploymentGenerationError, match="not a regular file"):
        deployment_plan.verify_plan_sources(plan)


def test_verify_plan_sources_rejects_policy_edits(tmp_path):
    write_deployment_policy(tmp_path)
    write_files(tmp_path, {"a.txt": "a"})
    plan = build_deployment_plan(tmp_path)

    write_deployment_policy(tmp_path, ["a.txt"])

    with pytest.raises(DeploymentGenerationError, match="deploy.toml changed"):
        deployment_plan.verify_plan_sources(plan)


def test_materialization_streams_exactly_the_frozen_digest(tmp_path):
    write_deployment_policy(tmp_path)
    write_files(tmp_path, {"a.txt": "planned"})
    plan = build_deployment_plan(tmp_path)
    target = tmp_path / "out" / "a.txt"

    deployment_plan.materialize_deployment_plan_entry(
        plan, _entry(plan, "a.txt"), target
    )

    assert target.read_text() == "planned"


def test_materialization_rejects_content_changed_after_planning(tmp_path):
    write_deployment_policy(tmp_path)
    source = tmp_path / "a.txt"
    source.write_text("planned")
    plan = build_deployment_plan(tmp_path)
    target = tmp_path / "out" / "a.txt"

    source.write_text("rewritten after planning")

    with pytest.raises(DeploymentGenerationError, match="content digest differs"):
        deployment_plan.materialize_deployment_plan_entry(
            plan, _entry(plan, "a.txt"), target
        )
    assert not target.exists()


def test_assembly_manifest_roundtrips_and_is_self_checking(tmp_path):
    staging = tmp_path / "assembly"
    write_files(staging, {"a.txt": "a", "bin/run.sh": "#!/bin/sh\n"})
    (staging / "bin" / "run.sh").chmod(0o755)
    manifest = deployment_plan.build_assembly_manifest(staging, generation="gen-1")

    text = manifest.to_json()
    assert manifest == deployment_plan.AssemblyManifest.from_json(text)
    assert "gen-1" in text
    assert manifest.root_digest.startswith("sha256:")

    payload = json.loads(text)
    payload["files"][0]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(DeploymentPlanError, match="root digest"):
        deployment_plan.AssemblyManifest.from_json(json.dumps(payload))


def test_assembly_manifest_path_sits_next_to_the_staging_tree(tmp_path):
    staging = tmp_path / "private" / "exp-uid"
    staging.mkdir(parents=True)

    assert deployment_plan.assembly_manifest_path(staging) == (
        tmp_path / "private" / deployment_plan.ASSEMBLY_MANIFEST_NAME
    )


def test_verify_assembly_directory_accepts_frozen_tree_and_declared_overlays(tmp_path):
    staging = tmp_path / "assembly"
    write_files(staging, {"a.txt": "a"})
    manifest = deployment_plan.build_assembly_manifest(staging, generation="gen-1")

    deployment_plan.verify_assembly_directory(staging, manifest)

    write_files(staging, {"Dockerfile": "FROM scratch\n"})
    with pytest.raises(DeploymentGenerationError, match="Dockerfile"):
        deployment_plan.verify_assembly_directory(staging, manifest)

    deployment_plan.verify_assembly_directory(staging, manifest, allow=("Dockerfile",))


def test_verify_assembly_directory_allows_declared_rewrite_of_frozen_file(tmp_path):
    staging = tmp_path / "assembly"
    write_files(staging, {"a.txt": "a", "Dockerfile": "FROM base:1\n"})
    manifest = deployment_plan.build_assembly_manifest(staging, generation="gen-1")

    # The framework rewrites a frozen destination after the freeze.
    (staging / "Dockerfile").write_text("FROM base:2\n# rewritten by framework\n")

    with pytest.raises(DeploymentGenerationError, match="Dockerfile"):
        deployment_plan.verify_assembly_directory(staging, manifest)
    deployment_plan.verify_assembly_directory(staging, manifest, allow=("Dockerfile",))


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda staging: (staging / "a.txt").write_text("changed"), id="content"
        ),
        pytest.param(lambda staging: (staging / "a.txt").unlink(), id="deleted"),
        pytest.param(
            lambda staging: write_files(staging, {"extra.txt": "extra"}), id="added"
        ),
        pytest.param(lambda staging: (staging / "a.txt").chmod(0o600), id="mode"),
    ],
)
def test_verify_assembly_directory_rejects_drift(tmp_path, mutate):
    staging = tmp_path / "assembly"
    write_files(staging, {"a.txt": "a", "keep.sh": "#!/bin/sh\n"})
    (staging / "keep.sh").chmod(0o755)
    manifest = deployment_plan.build_assembly_manifest(staging, generation="gen-1")

    mutate(staging)

    with pytest.raises(DeploymentGenerationError):
        deployment_plan.verify_assembly_directory(staging, manifest)


def test_verify_assembly_directory_rejects_symlink_in_isolated_tree(tmp_path):
    staging = tmp_path / "assembly"
    write_files(staging, {"a.txt": "a"})
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    manifest = deployment_plan.build_assembly_manifest(staging, generation="gen-1")

    (staging / "link.txt").symlink_to(outside)

    with pytest.raises(DeploymentPlanError, match="non-regular file"):
        deployment_plan.verify_assembly_directory(staging, manifest)


def _make_zip(root, base_name):
    return shutil.make_archive(str(base_name), "zip", root.parent, root.name)


def test_verify_zip_archive_members_match_manifest(tmp_path):
    staging = tmp_path / "assembly"
    write_files(staging, {"a.txt": "a", "d/b.txt": "b"})
    manifest = deployment_plan.build_assembly_manifest(staging, generation="gen-1")
    archive = _make_zip(staging, tmp_path / "bundle")

    deployment_plan.verify_zip_archive(archive, manifest)


def test_verify_zip_archive_strips_shared_root_directory(tmp_path):
    staging = tmp_path / "assembly"
    write_files(staging, {"a.txt": "a"})
    manifest = deployment_plan.build_assembly_manifest(staging, generation="gen-1")
    bundle = tmp_path / "prefixed.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("exp-uid/a.txt", "a")

    deployment_plan.verify_zip_archive(bundle, manifest)


def test_verify_zip_archive_rejects_added_changed_and_missing_members(tmp_path):
    staging = tmp_path / "assembly"
    write_files(staging, {"a.txt": "a", "d/b.txt": "b"})
    manifest = deployment_plan.build_assembly_manifest(staging, generation="gen-1")

    extra = tmp_path / "extra.zip"
    with zipfile.ZipFile(extra, "w") as archive:
        archive.writestr("a.txt", "a")
        archive.writestr("d/b.txt", "b")
        archive.writestr("rogue.txt", "sneaked in")
    with pytest.raises(DeploymentGenerationError, match="rogue.txt"):
        deployment_plan.verify_zip_archive(extra, manifest)

    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(changed, "w") as archive:
        archive.writestr("a.txt", "different bytes")
        archive.writestr("d/b.txt", "b")
    with pytest.raises(DeploymentGenerationError, match="a.txt"):
        deployment_plan.verify_zip_archive(changed, manifest)

    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("a.txt", "a")
    with pytest.raises(DeploymentGenerationError, match="b.txt.*missing"):
        deployment_plan.verify_zip_archive(missing, manifest)


def test_generation_label_is_a_stable_docker_label_name():
    assert deployment_plan.GENERATION_LABEL == "org.dallinger.deployment-generation"
