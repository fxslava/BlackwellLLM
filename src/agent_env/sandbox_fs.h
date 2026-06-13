// SandboxFs: a chroot-like filesystem facade for the autonomous agent.
//
// Every operation takes a path *relative to the sandbox root* and is guaranteed
// never to touch a byte outside that root. `..` traversal, absolute paths and
// (on resolve) symlinks that escape the root are all rejected *before* any I/O
// happens -- the containment check is purely lexical and runs first, so a
// hallucinated path like "../../../Windows/System32/foo" can never reach the
// host filesystem. Construct it on a directory you are willing to let the agent
// freely create, mutate and delete inside.
//
// This is the agent's only sanctioned door to the file system; it pairs with
// Subprocess (its only door to execution) and Workspace (which provisions the
// root directory itself).
#ifndef BLACKWELL_AGENT_ENV_SANDBOX_FS_H
#define BLACKWELL_AGENT_ENV_SANDBOX_FS_H

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace agent::env {

class SandboxFs {
public:
    // `root` is normalised to an absolute, symlink-resolved path up front so the
    // containment checks have a stable, canonical base to compare against.
    explicit SandboxFs(std::filesystem::path root);

    const std::filesystem::path& root() const { return root_; }

    // Resolve a sandbox-relative path to a real absolute path, or nullopt if it
    // would escape the sandbox. Performs no I/O on the target itself beyond a
    // symlink-resolution guard on existing ancestors. Public so callers can hand
    // a vetted absolute path to e.g. a subprocess working directory.
    std::optional<std::filesystem::path> resolve(std::string_view rel) const;

    bool exists(std::string_view rel) const;
    bool is_directory(std::string_view rel) const;

    // Read an entire file. nullopt if the path is unsafe or the file cannot be
    // read (missing, permissions, is a directory).
    std::optional<std::string> read(std::string_view rel) const;

    // Create or overwrite a file, creating any missing parent directories that
    // lie inside the sandbox. Returns false if the path is unsafe or the write
    // fails.
    bool write(std::string_view rel, std::string_view content);

    // Literal find/replace patch on an existing file: replaces the first
    // occurrence of `find` with `replace`. Returns false if the path is unsafe,
    // the file is missing, or `find` does not occur -- a patch that matches
    // nothing is reported as a failure rather than a silent no-op, so the agent
    // cannot believe it edited something it did not.
    bool patch(std::string_view rel, std::string_view find, std::string_view replace);

    // Create a directory (and parents) inside the sandbox.
    bool make_dirs(std::string_view rel);

    // Delete a file or directory tree inside the sandbox. Refuses to remove the
    // sandbox root itself. Returns false on an unsafe path; succeeds (true) if
    // the target is already absent.
    bool remove(std::string_view rel);

    // Non-recursive listing of a sandbox directory, returned as sandbox-relative
    // paths (forward-slashed). Empty if the path is unsafe or not a directory.
    std::vector<std::string> list(std::string_view rel = ".") const;

private:
    std::filesystem::path root_;
};

}  // namespace agent::env

#endif  // BLACKWELL_AGENT_ENV_SANDBOX_FS_H
