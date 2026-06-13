#include "sandbox_fs.h"

#include <fstream>
#include <system_error>

namespace agent::env {

namespace fs = std::filesystem;

namespace {

// True iff `p` is the same path as `base` or lives underneath it. Both paths are
// expected to be lexically normal and absolute. We use lexically_relative so the
// comparison is component-wise (not a string prefix), which avoids the classic
// "/sandbox-evil" being judged inside "/sandbox" bug.
bool within(const fs::path& base, const fs::path& p) {
    const fs::path rel = p.lexically_relative(base);
    if (rel.empty()) return false;
    // The first component being ".." means p climbs above base.
    const auto first = rel.begin();
    return first == rel.end() || first->native() != fs::path("..").native();
}

}  // namespace

SandboxFs::SandboxFs(fs::path root) {
    std::error_code ec;
    // weakly_canonical resolves symlinks/.. in whatever prefix already exists and
    // leaves the rest lexical, giving us a stable absolute base even if the
    // directory is created moments later.
    root_ = fs::weakly_canonical(root, ec);
    if (ec) root_ = fs::absolute(root).lexically_normal();
}

std::optional<fs::path> SandboxFs::resolve(std::string_view rel) const {
    const fs::path r(rel);

    // Only genuinely relative, root-less paths are admissible. This rejects
    // "C:\\...", "\\\\server\\share", "/etc/passwd" and bare drive-relative
    // forms before they can be joined onto the root.
    if (r.is_absolute() || r.has_root_name() || r.has_root_directory())
        return std::nullopt;

    // Primary, I/O-free guard: normalise the join and confirm lexical containment.
    const fs::path joined = (root_ / r).lexically_normal();
    if (!within(root_, joined)) return std::nullopt;

    // Secondary guard against symlinks inside the sandbox that point back out:
    // canonicalise the existing prefix and re-check. weakly_canonical never
    // throws here (error_code overload); on error we fall back to the lexical
    // result we already validated.
    std::error_code ec;
    const fs::path real = fs::weakly_canonical(joined, ec);
    if (!ec && !within(root_, real)) return std::nullopt;

    return joined;
}

bool SandboxFs::exists(std::string_view rel) const {
    const auto p = resolve(rel);
    if (!p) return false;
    std::error_code ec;
    return fs::exists(*p, ec);
}

bool SandboxFs::is_directory(std::string_view rel) const {
    const auto p = resolve(rel);
    if (!p) return false;
    std::error_code ec;
    return fs::is_directory(*p, ec);
}

std::optional<std::string> SandboxFs::read(std::string_view rel) const {
    const auto p = resolve(rel);
    if (!p) return std::nullopt;

    std::ifstream in(*p, std::ios::binary);
    if (!in) return std::nullopt;
    std::string data((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    if (in.bad()) return std::nullopt;
    return data;
}

bool SandboxFs::write(std::string_view rel, std::string_view content) {
    const auto p = resolve(rel);
    if (!p) return false;

    std::error_code ec;
    if (p->has_parent_path()) {
        fs::create_directories(p->parent_path(), ec);
        if (ec) return false;
    }

    std::ofstream out(*p, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return static_cast<bool>(out);
}

bool SandboxFs::patch(std::string_view rel, std::string_view find,
                      std::string_view replace) {
    if (find.empty()) return false;  // an empty needle is a programming error

    const auto current = read(rel);
    if (!current) return false;

    const auto pos = current->find(find);
    if (pos == std::string::npos) return false;  // surface the failed match

    std::string updated = *current;
    updated.replace(pos, find.size(), replace);
    return write(rel, updated);
}

bool SandboxFs::make_dirs(std::string_view rel) {
    const auto p = resolve(rel);
    if (!p) return false;
    std::error_code ec;
    fs::create_directories(*p, ec);
    // create_directories reports false (with no error) when the dir already
    // exists; treat "exists as a directory" as success.
    if (!ec) return true;
    return fs::is_directory(*p, ec);
}

bool SandboxFs::remove(std::string_view rel) {
    const auto p = resolve(rel);
    if (!p) return false;
    // Never let the agent nuke its own sandbox. Compare component-wise rather
    // than by string: "." normalises to the root *with* a trailing separator,
    // which is not byte-equal to root_ but is the same directory.
    const fs::path rel_to_root = p->lexically_relative(root_);
    if (rel_to_root.empty() || rel_to_root == fs::path(".")) return false;

    std::error_code ec;
    fs::remove_all(*p, ec);
    return !ec;
}

std::vector<std::string> SandboxFs::list(std::string_view rel) const {
    std::vector<std::string> out;
    const auto p = resolve(rel);
    if (!p) return out;

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(*p, ec)) {
        if (ec) break;
        out.push_back(entry.path().lexically_relative(root_).generic_string());
    }
    return out;
}

}  // namespace agent::env
