/* render.c — the C ABI, exercised end to end.
 *
 * Loads a preset, plays a standard MIDI file through the event API, writes a
 * 32-bit float stereo WAV. At 48 kHz its output is sample-for-sample what
 * `cargo run -p piano-emulator -- render out.wav in.mid` writes; that identity
 * is the point of the program and `ffi/tests/harness.rs` asserts it.
 *
 * It is deliberately not a library. Everything it does — parse an SMF, build a
 * tempo map, place events on blocks — a host does for us, and it is written
 * here only so that the C side of the boundary is exercised by C rather than by
 * Rust pretending to be C. Where it mirrors engine behaviour it says which
 * function it is mirroring, because the sample-exactness depends on the mirror
 * being honest:
 *
 *   - event time is `midi.rs`'s tick->second map, evaluated in double and then
 *     narrowed to float, exactly as `midly`'s is;
 *   - an event's frame is `RenderEvent::frame` — round-half-away-from-zero of
 *     `time_s * 48000` in *float*;
 *   - events are stable-sorted by tick and then stable-sorted by frame, as
 *     `midi::parse` and `render_to_buffer` do in that order;
 *   - an event takes effect at the start of the 128-frame block that contains
 *     its sample (`DECISIONS.md` 55), which is what the block loop below does;
 *   - the render is `duration = last_event + 4 s` long, truncated to a whole
 *     number of frames in float (`MidiPerformance::duration_s`).
 *
 * usage: render <preset.toml|-> <in.mid> <out.wav> [--rate HZ] [--queue]
 *   -         use the built-in default preset instead of loading a file
 *   --rate    host sample rate; anything but 48000 runs the boundary resampler
 *   --queue   post events through the SPSC queue (pe_post_event) rather than
 *             applying them on the audio thread (pe_event)
 *
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "piano_emulator.h"

/* The engine's block, and the grain every event quantises to. */
#define BLOCK 128
#define ENGINE_RATE 48000.0f

static void die(const char *what) {
    fprintf(stderr, "render: %s\n", what);
    exit(1);
}

/* ------------------------------------------------------------------ files */

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    unsigned char *buf = malloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (size > 0 && fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[size] = 0;
    fclose(f);
    *len = (size_t)size;
    return buf;
}

/* --------------------------------------------------------------- SMF read */

typedef struct {
    uint64_t tick;
    uint32_t order; /* file order, so the sorts below can be stable by hand */
    pe_event_t event;
    float time_s;
    size_t frame;
} timed_event;

typedef struct {
    uint64_t tick;
    double us_per_beat;
    uint32_t order;
} tempo_change;

/* (tick, seconds at that tick, seconds per tick from there on) — Clock in
 * `midi.rs`. */
typedef struct {
    uint64_t tick;
    double seconds;
    double rate;
} clock_segment;

typedef struct {
    const unsigned char *p;
    const unsigned char *end;
} cursor;

static uint32_t read_be(cursor *c, int bytes) {
    uint32_t v = 0;
    for (int i = 0; i < bytes; i++) {
        if (c->p >= c->end) {
            die("truncated MIDI file");
        }
        v = (v << 8) | *c->p++;
    }
    return v;
}

static uint32_t read_varlen(cursor *c) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        if (c->p >= c->end) {
            die("truncated variable-length quantity");
        }
        unsigned char b = *c->p++;
        v = (v << 7) | (b & 0x7f);
        if (!(b & 0x80)) {
            return v;
        }
    }
    die("over-long variable-length quantity");
    return 0;
}

/* Sorted insert-free stable merge sort over timed_event, keyed by a callback. */
static int by_tick(const timed_event *a, const timed_event *b) {
    if (a->tick != b->tick) {
        return a->tick < b->tick ? -1 : 1;
    }
    return a->order < b->order ? -1 : 1;
}

static int by_frame(const timed_event *a, const timed_event *b) {
    if (a->frame != b->frame) {
        return a->frame < b->frame ? -1 : 1;
    }
    return a->order < b->order ? -1 : 1;
}

static int cmp_tick(const void *a, const void *b) {
    return by_tick(a, b);
}

static int cmp_frame(const void *a, const void *b) {
    return by_frame(a, b);
}

/* `order` carries the original position, so `qsort` — which is not stable —
 * still reproduces Rust's stable sort exactly. */
static void stable_sort(timed_event *events, size_t n,
                        int (*cmp)(const void *, const void *)) {
    for (size_t i = 0; i < n; i++) {
        events[i].order = (uint32_t)i;
    }
    qsort(events, n, sizeof(timed_event), cmp);
}

/* MIDI 1.0 note range the instrument has keys for: A0..C8. */
static int playable(unsigned key) {
    return key >= 21 && key <= 108;
}

#define CC_SUSTAIN 64
#define CC_SOSTENUTO 66
#define CC_UNA_CORDA 67
#define SWITCH_THRESHOLD 64
#define DEFAULT_RELEASE_VELOCITY 64
#define DEFAULT_US_PER_BEAT 500000.0
#define RELEASE_TAIL_S 4.0f

/* `midi.rs::translate`. Returns 0 if the message is not one the engine has any
 * use for. */
static int translate(unsigned status, unsigned d1, unsigned d2, pe_event_t *out) {
    unsigned kind = status & 0xf0;
    memset(out, 0, sizeof(*out));
    if (kind == 0x90 && d2 > 0) {
        if (!playable(d1)) {
            return 0;
        }
        out->kind = PE_EVENT_NOTE_ON;
        out->key = d1;
        out->vel = d2;
        return 1;
    }
    if (kind == 0x80 || kind == 0x90) {
        if (!playable(d1)) {
            return 0;
        }
        out->kind = PE_EVENT_NOTE_OFF;
        out->key = d1;
        /* Zero is "this keyboard does not measure release velocity", not
         * "released infinitely slowly" — and a note-on at velocity 0 carries
         * none at all. */
        out->vel = (kind == 0x80 && d2 != 0) ? d2 : DEFAULT_RELEASE_VELOCITY;
        return 1;
    }
    if (kind == 0xb0) {
        int down = d2 >= SWITCH_THRESHOLD;
        switch (d1) {
        case CC_SUSTAIN:
            out->kind = PE_EVENT_SUSTAIN;
            out->value = (float)d2 / 127.0f;
            return 1;
        case CC_SOSTENUTO:
            out->kind = PE_EVENT_SOSTENUTO;
            out->value = down ? 1.0f : 0.0f;
            return 1;
        case CC_UNA_CORDA:
            out->kind = PE_EVENT_UNA_CORDA;
            out->value = down ? 1.0f : 0.0f;
            return 1;
        default:
            return 0;
        }
    }
    return 0;
}

static size_t parse_smf(const unsigned char *bytes, size_t len,
                        timed_event **out_events, float *out_last_s) {
    cursor c = {bytes, bytes + len};
    if (len < 14 || memcmp(bytes, "MThd", 4) != 0) {
        die("not a standard MIDI file");
    }
    c.p += 4;
    uint32_t header_len = read_be(&c, 4);
    uint32_t format = read_be(&c, 2);
    uint32_t ntracks = read_be(&c, 2);
    uint32_t division = read_be(&c, 2);
    if (format == 2) {
        die("unsupported MIDI file: format 2 (sequential tracks)");
    }
    if (division & 0x8000) {
        die("unsupported MIDI file: SMPTE timecode division");
    }
    if (division == 0) {
        die("unsupported MIDI file: zero ticks per beat");
    }
    c.p = bytes + 8 + header_len;

    size_t cap = 1024, n = 0;
    timed_event *events = malloc(cap * sizeof(timed_event));
    size_t tempo_cap = 64, tempo_n = 0;
    tempo_change *tempos = malloc(tempo_cap * sizeof(tempo_change));
    if (!events || !tempos) {
        die("out of memory");
    }

    for (uint32_t t = 0; t < ntracks && c.p + 8 <= c.end; t++) {
        if (memcmp(c.p, "MTrk", 4) != 0) {
            die("expected an MTrk chunk");
        }
        c.p += 4;
        uint32_t chunk_len = read_be(&c, 4);
        const unsigned char *track_end = c.p + chunk_len;
        if (track_end > c.end) {
            die("truncated track");
        }
        uint64_t tick = 0;
        unsigned running = 0;
        while (c.p < track_end) {
            tick += read_varlen(&c);
            unsigned status = *c.p;
            if (status & 0x80) {
                c.p++;
                if (status < 0xf0) {
                    running = status;
                }
            } else {
                if (!running) {
                    die("running status with no status byte");
                }
                status = running;
            }
            if (status == 0xff) {
                unsigned meta = *c.p++;
                uint32_t mlen = read_varlen(&c);
                if (meta == 0x51 && mlen == 3) {
                    double us = (double)((c.p[0] << 16) | (c.p[1] << 8) | c.p[2]);
                    if (tempo_n == tempo_cap) {
                        tempo_cap *= 2;
                        tempos = realloc(tempos, tempo_cap * sizeof(tempo_change));
                        if (!tempos) {
                            die("out of memory");
                        }
                    }
                    tempos[tempo_n].tick = tick;
                    tempos[tempo_n].us_per_beat = us;
                    tempos[tempo_n].order = (uint32_t)tempo_n;
                    tempo_n++;
                }
                c.p += mlen;
                continue;
            }
            if (status == 0xf0 || status == 0xf7) {
                uint32_t slen = read_varlen(&c);
                c.p += slen;
                continue;
            }
            unsigned high = status & 0xf0;
            unsigned d1 = 0, d2 = 0;
            int data_bytes = (high == 0xc0 || high == 0xd0) ? 1 : 2;
            d1 = *c.p++;
            if (data_bytes == 2) {
                d2 = *c.p++;
            }
            pe_event_t event;
            if (!translate(status, d1, d2, &event)) {
                continue;
            }
            if (n == cap) {
                cap *= 2;
                events = realloc(events, cap * sizeof(timed_event));
                if (!events) {
                    die("out of memory");
                }
            }
            events[n].tick = tick;
            events[n].event = event;
            n++;
        }
        c.p = track_end;
    }

    /* The tempo map: `midi.rs::Clock::new`, in double throughout. The changes
     * are stable-sorted by tick (insertion sort: there are never many), because
     * two tempo events on the same tick mean the last one in file order wins. */
    for (size_t i = 1; i < tempo_n; i++) {
        tempo_change key = tempos[i];
        size_t j = i;
        while (j > 0 && tempos[j - 1].tick > key.tick) {
            tempos[j] = tempos[j - 1];
            j--;
        }
        tempos[j] = key;
    }
    double ticks_per_beat = (double)division;
    clock_segment *segments = malloc((tempo_n + 1) * sizeof(clock_segment));
    if (!segments) {
        die("out of memory");
    }
    size_t nseg = 0;
    segments[nseg].tick = 0;
    segments[nseg].seconds = 0.0;
    segments[nseg].rate = DEFAULT_US_PER_BEAT / 1.0e6 / ticks_per_beat;
    nseg++;
    for (size_t i = 0; i < tempo_n; i++) {
        double rate = tempos[i].us_per_beat / 1.0e6 / ticks_per_beat;
        clock_segment last = segments[nseg - 1];
        double seconds =
            last.seconds + (double)(tempos[i].tick - last.tick) * last.rate;
        if (tempos[i].tick == last.tick) {
            segments[nseg - 1].tick = tempos[i].tick;
            segments[nseg - 1].seconds = seconds;
            segments[nseg - 1].rate = rate;
        } else {
            segments[nseg].tick = tempos[i].tick;
            segments[nseg].seconds = seconds;
            segments[nseg].rate = rate;
            nseg++;
        }
    }

    stable_sort(events, n, cmp_tick);
    for (size_t i = 0; i < n; i++) {
        size_t seg = 0;
        while (seg + 1 < nseg && segments[seg + 1].tick <= events[i].tick) {
            seg++;
        }
        double seconds = segments[seg].seconds +
                         (double)(events[i].tick - segments[seg].tick) *
                             segments[seg].rate;
        /* `as f32` in `midi.rs`, then `RenderEvent::frame`'s float round. */
        events[i].time_s = (float)seconds;
        float t = events[i].time_s > 0.0f ? events[i].time_s : 0.0f;
        events[i].frame = (size_t)roundf(t * ENGINE_RATE);
    }
    *out_last_s = n > 0 ? events[n - 1].time_s : 0.0f;
    stable_sort(events, n, cmp_frame);

    free(tempos);
    free(segments);
    *out_events = events;
    return n;
}

/* --------------------------------------------------------------- WAV write */

static void write_u32(FILE *f, uint32_t v) {
    unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8),
                          (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    fwrite(b, 1, 4, f);
}

static void write_u16(FILE *f, uint16_t v) {
    unsigned char b[2] = {(unsigned char)v, (unsigned char)(v >> 8)};
    fwrite(b, 1, 2, f);
}

static void write_wav(const char *path, const float *l, const float *r,
                      size_t frames, uint32_t rate) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        die("cannot open the output file");
    }
    uint32_t data_bytes = (uint32_t)(frames * 2 * sizeof(float));
    fwrite("RIFF", 1, 4, f);
    write_u32(f, 36 + data_bytes);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    write_u32(f, 16);
    write_u16(f, 3); /* IEEE float */
    write_u16(f, 2);
    write_u32(f, rate);
    write_u32(f, rate * 2 * (uint32_t)sizeof(float));
    write_u16(f, 2 * (uint16_t)sizeof(float));
    write_u16(f, 32);
    fwrite("data", 1, 4, f);
    write_u32(f, data_bytes);
    for (size_t i = 0; i < frames; i++) {
        fwrite(&l[i], sizeof(float), 1, f);
        fwrite(&r[i], sizeof(float), 1, f);
    }
    fclose(f);
}

/* --------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    const char *preset_path = NULL;
    const char *midi_path = NULL;
    const char *wav_path = NULL;
    double rate = 48000.0;
    int use_queue = 0;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rate") == 0 && i + 1 < argc) {
            rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--queue") == 0) {
            use_queue = 1;
        } else if (positional == 0) {
            preset_path = argv[i];
            positional++;
        } else if (positional == 1) {
            midi_path = argv[i];
            positional++;
        } else if (positional == 2) {
            wav_path = argv[i];
            positional++;
        } else {
            die("usage: render <preset.toml|-> <in.mid> <out.wav> "
                "[--rate HZ] [--queue]");
        }
    }
    if (positional != 3) {
        die("usage: render <preset.toml|-> <in.mid> <out.wav> "
            "[--rate HZ] [--queue]");
    }
    if (pe_abi_version() != PE_ABI_VERSION) {
        die("the library and this header disagree about the ABI version");
    }

    size_t midi_len = 0;
    unsigned char *midi_bytes = read_file(midi_path, &midi_len);
    if (!midi_bytes) {
        die("cannot read the MIDI file");
    }
    timed_event *events = NULL;
    float last_event_s = 0.0f;
    size_t nevents = parse_smf(midi_bytes, midi_len, &events, &last_event_s);
    free(midi_bytes);

    /* `MidiPerformance::duration_s` and `render_to_buffer`'s frame count. */
    float duration_s = last_event_s + RELEASE_TAIL_S;
    float engine_frames_f = duration_s * ENGINE_RATE;
    size_t engine_frames = (size_t)(engine_frames_f > 0.0f ? engine_frames_f : 0.0f);
    /* At the engine's own rate the two counts are the same number; at any other
     * host rate the file is as long in seconds, not in frames. */
    size_t out_frames =
        (size_t)((double)engine_frames * rate / (double)ENGINE_RATE);

    pe_engine *engine = pe_create(rate, BLOCK);
    if (!engine) {
        die("pe_create refused the sample rate or block size");
    }

    if (strcmp(preset_path, "-") != 0) {
        size_t toml_len = 0;
        unsigned char *toml = read_file(preset_path, &toml_len);
        if (!toml) {
            die("cannot read the preset");
        }
        if (pe_load_preset_toml(engine, (const char *)toml, toml_len) != PE_OK) {
            fprintf(stderr, "render: %s\n", pe_last_error(engine));
            exit(1);
        }
        free(toml);
    }

    float *left = calloc(out_frames ? out_frames : 1, sizeof(float));
    float *right = calloc(out_frames ? out_frames : 1, sizeof(float));
    if (!left || !right) {
        die("out of memory");
    }

    /* The block loop of `render_to_buffer`, event for event. The host rate only
     * changes how many frames come out of each block, never when an event
     * lands: events are placed on the engine's clock, which is the only clock
     * the instrument has. */
    size_t next = 0;
    size_t engine_pos = 0;
    size_t out_pos = 0;
    while (engine_pos < engine_frames) {
        size_t end = engine_pos + BLOCK;
        if (end > engine_frames) {
            end = engine_frames;
        }
        while (next < nevents && events[next].frame < end) {
            if (use_queue) {
                if (!pe_post_event(engine, events[next].event)) {
                    die("the event queue was full");
                }
            } else {
                pe_event(engine, events[next].event);
            }
            next++;
        }
        /* Host frames for this engine block: the running conversion, so
         * rounding never accumulates. */
        size_t want = (size_t)((double)end * rate / (double)ENGINE_RATE);
        if (want > out_frames) {
            want = out_frames;
        }
        if (want > out_pos) {
            pe_render(engine, left + out_pos, right + out_pos,
                      (uint32_t)(want - out_pos));
            out_pos = want;
        }
        engine_pos = end;
    }
    if (out_pos < out_frames) {
        pe_render(engine, left + out_pos, right + out_pos,
                  (uint32_t)(out_frames - out_pos));
    }

    write_wav(wav_path, left, right, out_frames, (uint32_t)rate);
    fprintf(stderr, "render: wrote %s (%zu frames at %.0f Hz, %zu events)\n",
            wav_path, out_frames, rate, nevents);

    free(left);
    free(right);
    free(events);
    pe_destroy(engine);
    return 0;
}
