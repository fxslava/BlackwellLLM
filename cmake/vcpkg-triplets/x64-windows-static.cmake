# =============================================================================
# x64-windows-static, OVERRIDDEN to pin the Visual Studio instance.
#
# Used via --overlay-triplets so it KEEPS THE STOCK TRIPLET NAME. That is the
# point: the install root stays vcpkg/installed/x64-windows-static, so
# cmake/WebRtcAec3.cmake and the x64-*-aec3 presets need no separate path for a
# "special" triplet, and a tree configured against the stock triplet keeps
# working unchanged.
#
# =============================================================================
# WHY THE VS INSTANCE HAS TO BE PINNED
# =============================================================================
# vcpkg picks the NEWEST Visual Studio instance it can find, and this repo does
# not: CMakePresets pins "Visual Studio 17 2022". On a box with both VS 2022 and
# VS 18 installed, those two policies disagree, and the disagreement is invisible
# until the very end of the link:
#
#   vcpkg -> VS 18   -> MSVC 14.50 (v145) -> webrtc.lib, absl_strings.lib
#   repo  -> VS 2022 -> MSVC 14.4x (v143) -> everything else
#
# MSVC's separately-compiled STL grows new entry points between releases, so the
# 14.50-built objects call into things a v143 libcpmt.lib has never heard of:
#
#   __std_find_first_of_trivial_pos_1
#   __std_min_element_f / __std_max_element_f / __std_minmax_element_f
#
# and the build dies with LNK2001 on symbols nobody in this repo wrote. Linking
# the newer CRT instead would be mixing toolsets across the v143 -> v145 major
# boundary, which is not a compatibility guarantee worth relying on. Building the
# dependency with the toolset that will link it is the version that has no
# caveat.
#
# So: the dependency follows the project, not the other way round. If this repo
# ever moves off VS 2022, this path moves with it.
#
# =============================================================================
# IF VCPKG CANNOT FIND THIS PATH
# =============================================================================
# vcpkg silently falls back to its own instance search -- there is no error for a
# VCPKG_VISUAL_STUDIO_PATH that does not exist. The symptom is the LNK2001 list
# above, coming back after a rebuild that looked like it worked. Check the path
# below against the box before assuming the port is at fault.
# =============================================================================

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE static)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_PROVIDED_FORTRAN ON)

# v143 is the toolset the presets' "Visual Studio 17 2022" generator selects.
set(VCPKG_PLATFORM_TOOLSET v143)

# BACKSLASHES, and they are load-bearing. vcpkg compares this string against the
# instance root paths it discovers, which come back from the VS setup registry in
# native Windows form; a forward-slash spelling does not match any of them and
# fails with "Unable to find a valid Visual Studio instance" while listing the
# very instance it just refused.
set(VCPKG_VISUAL_STUDIO_PATH "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community")
