#pragma once
// =============================================================================
// cloud/simdjson_external.hpp — the ONE place <simdjson.h> is included.
//
// WHY THIS FILE EXISTS. The root's zero-warning policy quarantines third-party
// headers two ways at once (`/external:anglebrackets` + `/external:W0`), and
// simdjson satisfies both -- it is an angle-bracket include AND it sits under a
// `/external:I` directory. C4706 ("assignment within conditional expression")
// escapes anyway, from simdjson's `simdjson_inline` bodies: MSVC analyses those
// when our code first references them, and the warning state it applies is the
// one at the REFERENCE, which is first-party and therefore /W4 /WX.
//
// So the header slips past `/external`, which is the exact case the policy
// sanctions a scoped `#pragma warning(push/disable/pop)` for (skill
// `compiler-hygiene`). It is wrapped HERE, once, rather than at each of the two
// decoders: a third consumer must not have to rediscover this.
//
// The suppression is C4706 ONLY, and it is simdjson's idiom, not a defect --
// `if ((error = it.skip_child()))` is how that library propagates error codes.
// Nothing first-party is silenced: our own code still compiles at /W4 /WX.
// =============================================================================
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4706)  // assignment within conditional expression
#endif

#include <simdjson.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
