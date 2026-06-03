
#ifndef NVCOMP_EXPORT_H
#define NVCOMP_EXPORT_H

#ifdef NVCOMP_STATIC_DEFINE
#  define NVCOMP_EXPORT
#  define NVCOMP_NO_EXPORT
#else
#  ifndef NVCOMP_EXPORT
#    ifdef nvcomp_EXPORTS
        /* We are building this library */
#      define NVCOMP_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define NVCOMP_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef NVCOMP_NO_EXPORT
#    define NVCOMP_NO_EXPORT 
#  endif
#endif

#ifndef NVCOMP_DEPRECATED
#  define NVCOMP_DEPRECATED __declspec(deprecated)
#endif

#ifndef NVCOMP_DEPRECATED_EXPORT
#  define NVCOMP_DEPRECATED_EXPORT NVCOMP_EXPORT NVCOMP_DEPRECATED
#endif

#ifndef NVCOMP_DEPRECATED_NO_EXPORT
#  define NVCOMP_DEPRECATED_NO_EXPORT NVCOMP_NO_EXPORT NVCOMP_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef NVCOMP_NO_DEPRECATED
#    define NVCOMP_NO_DEPRECATED
#  endif
#endif

#endif /* NVCOMP_EXPORT_H */
