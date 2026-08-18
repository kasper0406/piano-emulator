/* pe_shim.h — the render block's state, in C, because it has to be POD.
 *
 * The AUv3's `internalRenderBlock` runs on the audio thread under the same
 * contract `piano_emulator.h` states for `pe_render`: no allocation, no locks,
 * no syscalls. Swift gives us no way to *say* that, and two Swift constructs
 * quietly break it — ARC traffic on a captured class reference is unbounded
 * retain/release on the audio thread, and a Swift `Atomic` (Synchronization,
 * macOS 15+) would raise the deployment target for six words of shared state.
 *
 * So the state the render block touches is a plain C struct, allocated once in
 * `allocateRenderResources()` and reached through a raw pointer the block
 * captures **by value**. Every field is read and written through the accessors
 * below rather than through Swift's view of the struct, for two reasons: a C
 * `_Atomic` field has no Swift spelling at all, and taking `&state.pointee.x`
 * in Swift is an inout access whose address Swift is not required to make the
 * real one. Passing the whole `pe_render_state *` to a `static inline` accessor
 * has neither problem and compiles to the single instruction it names.
 *
 * Nothing here is compiled into a library: every function is `static inline`.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PE_SHIM_H
#define PE_SHIM_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Parameter slots, in `AUParameterTree` address order. */
#define PE_PARAM_SUSTAIN 0
#define PE_PARAM_SOSTENUTO 1
#define PE_PARAM_UNA_CORDA 2
#define PE_PARAM_OUTPUT_TRIM 3
#define PE_PARAM_COUNT 4

/* The engine's own block. An event takes effect at the start of the 128-frame
 * block that contains its sample (`DECISIONS.md` 55), and the render block
 * splits its rendering on that grid so that a host's buffer size cannot move an
 * onset. Same number as `ffi/harness/render.c`'s `BLOCK`, for the same reason:
 * the two have to agree for the parity harness to be sample-exact. */
#define PE_BLOCK 128

/* Everything the render block reads or writes.
 *
 * Ownership, field by field:
 *
 *   engine    main -> audio. Published with a release store when the preset
 *             changes; the audio thread loads it once per render call. An
 *             engine is never destroyed while render resources are allocated —
 *             the replaced ones are held and freed in
 *             `deallocateRenderResources` — so the pointer the audio thread
 *             loads is always live.
 *   params    any -> audio. Float bits, written by the parameter tree's value
 *             observer, which a host may call from any thread.
 *   voices,
 *   peak_*    audio -> any. Metering, read by a UI at its own leisure; a stale
 *             read is a wrong pixel and nothing else, so these are relaxed.
 *   applied,
 *   frames    audio only. No atomics: nothing else ever touches them.
 */
typedef struct {
    _Atomic(void *) engine;
    _Atomic uint32_t params[PE_PARAM_COUNT];
    _Atomic uint32_t voices;
    _Atomic uint32_t peak_left;
    _Atomic uint32_t peak_right;
    /* Last parameter values pushed into the engine, so a render call only
     * emits an event for a parameter that actually moved. */
    float applied[PE_PARAM_COUNT];
    /* Engine frames rendered since `allocateRenderResources`. This is what
     * keeps an event on the 128-frame grid across host buffer boundaries. */
    uint64_t frames;
    /* Scratch output, used only when the host hands us a null buffer list. */
    float *scratch_left;
    float *scratch_right;
    uint32_t scratch_frames;
} pe_render_state;

/* A float carried through an atomic word. `memcpy` and not a union cast,
 * because the union cast is the one spelling of this that is undefined. */
static inline float pe_bits_to_float(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static inline uint32_t pe_float_to_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    return bits;
}

static inline void pe_state_init(pe_render_state *s) {
    memset(s, 0, sizeof *s);
    atomic_store_explicit(&s->engine, (void *)0, memory_order_release);
    for (int i = 0; i < PE_PARAM_COUNT; i++) {
        atomic_store_explicit(&s->params[i], pe_float_to_bits(0.0f),
                              memory_order_relaxed);
        s->applied[i] = 0.0f;
    }
    /* The trim starts at unity, not at zero: it is a gain, and its parameter
     * is in dB where zero is unity. Both halves say the same thing. */
    atomic_store_explicit(&s->params[PE_PARAM_OUTPUT_TRIM],
                          pe_float_to_bits(0.0f), memory_order_relaxed);
}

static inline void *pe_state_engine(const pe_render_state *s) {
    return atomic_load_explicit(&s->engine, memory_order_acquire);
}

static inline void pe_state_set_engine(pe_render_state *s, void *engine) {
    atomic_store_explicit(&s->engine, engine, memory_order_release);
}

static inline float pe_state_param(const pe_render_state *s, int index) {
    return pe_bits_to_float(
        atomic_load_explicit(&s->params[index], memory_order_relaxed));
}

static inline void pe_state_set_param(pe_render_state *s, int index,
                                      float value) {
    atomic_store_explicit(&s->params[index], pe_float_to_bits(value),
                          memory_order_relaxed);
}

static inline float pe_state_applied(const pe_render_state *s, int index) {
    return s->applied[index];
}

static inline void pe_state_set_applied(pe_render_state *s, int index,
                                        float value) {
    s->applied[index] = value;
}

static inline uint64_t pe_state_frames(const pe_render_state *s) {
    return s->frames;
}

static inline void pe_state_set_frames(pe_render_state *s, uint64_t frames) {
    s->frames = frames;
}

static inline void pe_state_publish_meter(pe_render_state *s, uint32_t voices,
                                          float peak_left, float peak_right) {
    atomic_store_explicit(&s->voices, voices, memory_order_relaxed);
    atomic_store_explicit(&s->peak_left, pe_float_to_bits(peak_left),
                          memory_order_relaxed);
    atomic_store_explicit(&s->peak_right, pe_float_to_bits(peak_right),
                          memory_order_relaxed);
}

static inline uint32_t pe_state_voices(const pe_render_state *s) {
    return atomic_load_explicit(&s->voices, memory_order_relaxed);
}

static inline float pe_state_peak_left(const pe_render_state *s) {
    return pe_bits_to_float(
        atomic_load_explicit(&s->peak_left, memory_order_relaxed));
}

static inline float pe_state_peak_right(const pe_render_state *s) {
    return pe_bits_to_float(
        atomic_load_explicit(&s->peak_right, memory_order_relaxed));
}

static inline void pe_state_set_scratch(pe_render_state *s, float *left,
                                        float *right, uint32_t frames) {
    s->scratch_left = left;
    s->scratch_right = right;
    s->scratch_frames = frames;
}

static inline float *pe_state_scratch_left(const pe_render_state *s) {
    return s->scratch_left;
}

static inline float *pe_state_scratch_right(const pe_render_state *s) {
    return s->scratch_right;
}

static inline uint32_t pe_state_scratch_frames(const pe_render_state *s) {
    return s->scratch_frames;
}

#endif /* PE_SHIM_H */
