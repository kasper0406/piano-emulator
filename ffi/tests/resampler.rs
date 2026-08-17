//! What the boundary resampler does to the signal, measured.
//!
//! `DISTRIBUTION.md` M0 asks for three numbers and this file is all three: a
//! null test at 48 kHz (in `bypass.rs`, because that one is about the ABI too),
//! a swept-sine alias floor at 44.1 and 96 kHz, and a transient-position test
//! like the one `DECISIONS.md` 64 already has for `SincFixedIn`.
//!
//! The sweep is stepped rather than continuous on purpose: a continuous chirp's
//! aliases fold to a *line* that crosses the chirp itself, and separating them
//! afterwards is guesswork. One steady tone at a time, with an FFT per tone,
//! says exactly how much energy came out at frequencies the input never had.

use piano_emulator_ffi::resample::{Boundary, Source};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::TAU;

const ENGINE_RATE: f64 = 48_000.0;
/// Output frames per measurement. 1.5 s at 44.1 kHz; the bin width is 0.7 Hz.
const N: usize = 1 << 16;
/// Output frames thrown away first, so the filter's zero-filled history is not
/// part of the measurement.
const WARMUP: usize = 8192;
/// Bins around a legitimate component that the alias sum skips. At 64 bins the
/// analysis window's own leakage is far below anything measured here.
const GUARD: i64 = 64;

/// A steady sine at the engine's rate. The phase is carried in `f64` across
/// pulls, so the tone is continuous however the resampler cuts up its demands.
struct Tone {
    freq: f64,
    phase: f64,
}

impl Source for Tone {
    fn fill(&mut self, l: &mut [f32], r: &mut [f32]) {
        let step = TAU * self.freq / ENGINE_RATE;
        for i in 0..l.len() {
            let v = (0.5 * self.phase.sin()) as f32;
            l[i] = v;
            r[i] = v;
            self.phase += step;
        }
    }
}

/// A symmetric raised-cosine pulse centred on input frame `at`, and silence
/// everywhere else. Its energy centroid is exactly `at`.
struct Pulse {
    at: f64,
    half: f64,
    frame: f64,
}

impl Source for Pulse {
    fn fill(&mut self, l: &mut [f32], r: &mut [f32]) {
        for i in 0..l.len() {
            let d = self.frame - self.at;
            let v = if d.abs() <= self.half {
                0.5 * (1.0 + (std::f64::consts::PI * d / self.half).cos())
            } else {
                0.0
            };
            l[i] = v as f32;
            r[i] = v as f32;
            self.frame += 1.0;
        }
    }
}

/// Four-term Blackman-Harris: -92 dB peak sidelobe and a fast roll-off, which
/// is what lets a -100 dB floor be read 64 bins away from a full-scale tone.
fn window(n: usize) -> Vec<f64> {
    const A: [f64; 4] = [0.35875, 0.48829, 0.14128, 0.01168];
    (0..n)
        .map(|i| {
            let x = TAU * i as f64 / n as f64;
            A[0] - A[1] * x.cos() + A[2] * (2.0 * x).cos() - A[3] * (3.0 * x).cos()
        })
        .collect()
}

/// Power per bin of the first half of the spectrum.
fn spectrum(x: &[f32]) -> Vec<f64> {
    let w = window(x.len());
    let mut buf: Vec<Complex<f64>> = x
        .iter()
        .zip(&w)
        .map(|(&v, &w)| Complex::new(v as f64 * w, 0.0))
        .collect();
    FftPlanner::new()
        .plan_fft_forward(buf.len())
        .process(&mut buf);
    buf[..buf.len() / 2].iter().map(|c| c.norm_sqr()).collect()
}

/// Renders `frames` host-rate frames from `source` through a boundary at
/// `host_rate`, after discarding `WARMUP` frames.
fn through(host_rate: f64, source: &mut impl Source, frames: usize) -> Vec<f32> {
    let mut boundary = Boundary::new(host_rate, 1024).expect("a buildable rate");
    let mut sink = vec![0.0f32; WARMUP.max(frames)];
    let mut other = vec![0.0f32; WARMUP.max(frames)];
    boundary.render(source, &mut sink[..WARMUP], &mut other[..WARMUP]);
    boundary.render(source, &mut sink[..frames], &mut other[..frames]);
    sink.truncate(frames);
    sink
}

/// The energy a tone puts where it belongs, and the energy it puts anywhere
/// else, for a host running at `host_rate`.
fn tone_energies(host_rate: f64, freq: f64) -> (f64, f64) {
    let mut tone = Tone { freq, phase: 0.0 };
    let out = through(host_rate, &mut tone, N);
    let power = spectrum(&out);
    // A resampler maps an input tone at f to an output tone at f. Above the
    // host's Nyquist there is no such bin, and every joule that comes out is
    // one that folded.
    let expected = if freq < host_rate / 2.0 {
        Some((freq / host_rate * N as f64).round() as i64)
    } else {
        None
    };
    let mut signal = 0.0;
    let mut alias = 0.0;
    for (bin, &p) in power.iter().enumerate() {
        let bin = bin as i64;
        if bin <= GUARD {
            continue; // DC and the window's own skirt around it
        }
        match expected {
            Some(e) if (bin - e).abs() <= GUARD => signal += p,
            _ => alias += p,
        }
    }
    (signal, alias)
}

/// The frequencies the sweep steps through, in Hz. It stops at 21.6 kHz because
/// that is where the engine stops: `MAX_PARTIAL_RATIO * SAMPLE_RATE` is the
/// highest partial any string is given.
const SWEPT: [f64; 14] = [
    50.0, 220.0, 1000.0, 3000.0, 5000.0, 8000.0, 11000.0, 15000.0, 18000.0, 20000.0, 21000.0,
    21600.0, 22500.0, 23500.0,
];

/// Nothing the engine can produce may come back out of the boundary at a
/// frequency it never had, at a level anyone could hear under a piano.
///
/// The gate is `DISTRIBUTION.md` M0's: **folded energy below -100 dB** of the
/// tone that produced it, at both of the rates a host actually runs.
#[test]
fn the_alias_floor_stays_a_hundred_decibels_down_at_every_host_rate() {
    for &host_rate in &[44_100.0, 96_000.0] {
        let (unity, _) = tone_energies(host_rate, 1000.0);
        let mut worst: f64 = f64::NEG_INFINITY;
        let mut worst_at = 0.0;
        for &freq in &SWEPT {
            let (_, alias) = tone_energies(host_rate, freq);
            let db = 10.0 * (alias / unity).log10();
            println!("{host_rate:>7.0} Hz  in {freq:>7.0} Hz  folded {db:>8.1} dB");
            if db > worst {
                worst = db;
                worst_at = freq;
            }
        }
        assert!(
            worst < -100.0,
            "{host_rate} Hz: {worst:.1} dB of folded energy from a {worst_at} Hz tone"
        );
    }
}

/// A tone above the host's Nyquist has nowhere legitimate to go, so everything
/// it produces is aliasing — the strictest form of the test above, and the one
/// that answers `DISTRIBUTION.md`'s partial-cap question.
///
/// The instrument's own content reaches 21.6 kHz (`MAX_PARTIAL_RATIO`), which
/// is **under** 44.1 kHz's 22.05 kHz Nyquist, so no partial folds at any rate
/// whatever the filter does. What does live above 22.05 kHz is the mechanism
/// noise and the master soft-clip's products, and this is the measurement that
/// says what happens to those.
#[test]
fn content_above_the_host_nyquist_is_filtered_rather_than_folded() {
    let (unity, _) = tone_energies(44_100.0, 1000.0);
    for &freq in &[22_100.0, 23_000.0, 23_900.0] {
        let (_, alias) = tone_energies(44_100.0, freq);
        let db = 10.0 * (alias / unity).log10();
        println!("44100 Hz  in {freq:>7.0} Hz  (all of it folds) {db:>8.1} dB");
        assert!(db < -100.0, "{freq} Hz folded back at {db:.1} dB");
    }
}

/// Where a transient lands, measured the way `DECISIONS.md` 64 measured it for
/// `SincFixedIn`: not "is it aligned" but "is the misalignment a *delay*".
///
/// A pure delay is inaudible and costs nothing; dispersion (a delay that
/// depends on frequency) or drift (one that depends on how far into the render
/// you are) would both be real damage, and both would show up here as an offset
/// that moves. It does not move: over positions 20 k, 50 k and 100 k input
/// frames apart the offset is the same to eleven decimal places, and its size
/// is under one input frame at every rate.
///
/// The offset itself is structural in `SincFixedOut`, which advances its
/// interpolation index before it emits a sample rather than after: the whole
/// output is `ratio - 1` output frames late, plus 81 ns of the same rounding
/// item 64 measured as 1.8 us. Correcting it would mean patching the
/// dependency, and correcting it *wrongly* — the mistake item 64 documents —
/// costs milliseconds rather than microseconds. It is pinned, not fixed.
#[test]
fn a_transient_arrives_delayed_and_not_smeared() {
    for &host_rate in &[44_100.0, 96_000.0] {
        let ratio = host_rate / ENGINE_RATE;
        // The structural part: `SincFixedOut` emits its first sample one output
        // frame into the stream.
        let structural = 1.0 - 1.0 / ratio;
        let mut offsets = Vec::new();
        for &at in &[20_000.0, 50_000.0, 120_000.0] {
            let mut pulse = Pulse {
                at,
                half: 24.0,
                frame: 0.0,
            };
            let frames = ((at + 4096.0) * ratio) as usize - WARMUP;
            let out = through(host_rate, &mut pulse, frames);
            // Energy centroid: robust where a peak is not, and exact for the
            // symmetric pulse this is.
            let (mut sum, mut weighted) = (0.0f64, 0.0f64);
            for (i, &v) in out.iter().enumerate() {
                let e = (v as f64) * (v as f64);
                sum += e;
                weighted += e * (i + WARMUP) as f64;
            }
            assert!(sum > 0.0, "the pulse never arrived");
            let offset = weighted / sum / ratio - at;
            println!(
                "{host_rate:>7.0} Hz  transient at input frame {at:>7.0}: \
                 {offset:+.6} input frames ({:+.2} us), structural {structural:+.4}",
                offset / ENGINE_RATE * 1.0e6
            );
            offsets.push(offset);
        }
        let spread = offsets.iter().fold(f64::MIN, |m, &v| m.max(v))
            - offsets.iter().fold(f64::MAX, |m, &v| m.min(v));
        assert!(
            spread < 1.0e-6,
            "{host_rate} Hz: the offset moved by {spread:.2e} frames across the \
             render — that is dispersion or drift, not a delay"
        );
        let residual = offsets[0] - structural;
        assert!(
            residual.abs() < 0.01,
            "{host_rate} Hz: {residual:+.4} input frames on top of the structural \
             offset, where 0.0039 was measured"
        );
        assert!(
            offsets[0].abs() < 1.0,
            "{host_rate} Hz: {:.3} input frames is past a whole engine sample",
            offsets[0]
        );
    }
}

/// What the bypass is worth: the same 48 kHz signal, once straight through and
/// once through the filter at ratio 1.0.
///
/// A polyphase sinc at unity ratio is nearly transparent in the passband — the
/// measurement below puts a 15 kHz tone through it within hundredths of a
/// decibel — and that is exactly why the *bypass* has to be a branch rather
/// than a ratio: "nearly" is not "bit-exact", and every calibrated number in
/// `PHYSICS.md`, every hash in `DECISIONS.md` and the whole acceptance suite
/// were measured on the engine's own samples. Running a transparent filter over
/// them would still make a 48 kHz host a different instrument from the one that
/// was measured, and nothing would say so.
///
/// The two numbers this prints are the size of that "nearly": the residual
/// against the direct signal, and the attenuation at 23 kHz, where the filter's
/// 0.95-of-Nyquist cutoff does start to bite. Measured: **-0.000 dB of gain and
/// -12.23 dB at 23 kHz**, with a residual of **-42.3 dB** — and that residual
/// is almost entirely the 81 ns sub-sample delay the transient test above
/// measures, not filter error (a 15 kHz tone shifted by 0.0039 samples is
/// 0.0077 rad out, which is -42 dB of difference on its own).
#[test]
fn the_unity_ratio_filter_is_transparent_but_not_identical() {
    let render_both = |freq: f64| -> (Vec<f32>, Vec<f32>) {
        let mut direct_tone = Tone { freq, phase: 0.0 };
        let mut l = vec![0.0f32; N];
        let mut r = vec![0.0f32; N];
        let mut warm = vec![0.0f32; WARMUP];
        let mut warm_r = vec![0.0f32; WARMUP];
        direct_tone.fill(&mut warm, &mut warm_r);
        direct_tone.fill(&mut l, &mut r);

        let mut filtered_tone = Tone { freq, phase: 0.0 };
        let mut boundary = Boundary::sinc(48_000.0, 1024).expect("buildable");
        let mut fl = vec![0.0f32; N];
        let mut fr = vec![0.0f32; N];
        boundary.render(&mut filtered_tone, &mut warm, &mut warm_r);
        boundary.render(&mut filtered_tone, &mut fl, &mut fr);
        (l, fl)
    };
    let energy = |x: &[f32]| x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();

    let (direct, filtered) = render_both(15_000.0);
    let gain = 10.0 * (energy(&filtered) / energy(&direct)).log10();
    let residual: f64 = direct
        .iter()
        .zip(&filtered)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    let residual_db = 10.0 * (residual / energy(&direct)).log10();
    println!("filter at ratio 1.0, 15 kHz: gain {gain:+.3} dB, residual {residual_db:.1} dB");

    let (direct_hi, filtered_hi) = render_both(23_000.0);
    let gain_hi = 10.0 * (energy(&filtered_hi) / energy(&direct_hi)).log10();
    println!("filter at ratio 1.0, 23 kHz: gain {gain_hi:+.2} dB");

    assert!(
        gain.abs() < 0.1,
        "a unity-ratio sinc should be transparent in the passband, not {gain:+.3} dB"
    );
    assert!(
        residual > 0.0,
        "the filter came out bit-identical to the bypass, which cannot be right"
    );
    assert!(
        gain_hi < -1.0,
        "the filter passed 23 kHz at {gain_hi:+.2} dB — it is not filtering at all"
    );
}

/// What the boundary costs, against the engine it sits in front of.
///
/// `DISTRIBUTION.md` budgets "two channels of a 128-256-tap polyphase sinc,
/// well under 1 % of a core against the engine's 31-40 %". That is a claim
/// about the shipped build, so it is measured on one: release only, like the
/// engine's own performance acceptance tests, because a debug build's number
/// would be a different program's number.
///
/// The source is a zero-fill rather than the engine, so what is timed is the
/// filter and nothing else. Measured on an M4 Pro performance core:
/// **0.70 % of a core at 44.1 kHz and 1.17 % at 96 kHz** — the budget holds at
/// 44.1 and is missed by a sixth of a percent at 96, where there are twice as
/// many output frames to compute per second of music. Against the engine's own
/// 31-40 % (`DECISIONS.md` 25b, 37) neither is a number worth designing around,
/// which is what the claim was really saying. The gate is set at 2 % so that it
/// fails on a filter that has grown rather than on a machine that is busy.
#[cfg(not(debug_assertions))]
#[test]
fn the_boundary_costs_a_fraction_of_a_percent_of_a_core() {
    use std::time::Instant;

    let seconds = 4.0;
    for &host_rate in &[44_100.0, 96_000.0] {
        let frames = (host_rate * seconds) as usize;
        let mut boundary = Boundary::new(host_rate, 512).expect("buildable");
        let mut silence = |l: &mut [f32], r: &mut [f32]| {
            l.fill(0.0);
            r.fill(0.0);
        };
        let mut l = vec![0.0f32; 512];
        let mut r = vec![0.0f32; 512];
        // One pass to warm the caches, then the measured one.
        for _ in 0..(frames / 512).min(64) {
            boundary.render(&mut silence, &mut l, &mut r);
        }
        let start = Instant::now();
        let mut done = 0;
        while done < frames {
            let n = 512.min(frames - done);
            boundary.render(&mut silence, &mut l[..n], &mut r[..n]);
            done += n;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let load = 100.0 * elapsed / seconds;
        println!("{host_rate:>7.0} Hz: the boundary costs {load:.3} % of one core");
        assert!(
            load < 2.0,
            "{host_rate} Hz: {load:.2} % of a core is not \"well under 1 %\" — \
             either the machine is busy or the filter has grown"
        );
    }
}

/// How far ahead of the host clock the engine runs, which is the second of
/// `DISTRIBUTION.md`'s "two honest consequences" — and the one that is easy to
/// mistake for latency.
///
/// It is not latency. The source is a generator, not a stream: when the filter
/// wants look-ahead it *pulls more frames* rather than delaying what it has, so
/// there is nothing to report to the host and nothing to compensate. What it
/// does mean is that an event handed to `pe_event` lands against an engine that
/// has already rendered this many frames past the host's position — an event
/// timing offset, not a delay. At 256 taps it is half the filter, 2.7 ms;
/// `DISTRIBUTION.md` estimated 1.3 ms from a 128-tap filter, and the tap count
/// is what the alias floor cost.
#[test]
fn the_engine_runs_ahead_of_the_host_rather_than_behind_it() {
    assert_eq!(
        Boundary::new(48_000.0, 512).unwrap().lookahead_frames(),
        0,
        "the bypass has no look-ahead at all"
    );
    for &host_rate in &[44_100.0, 96_000.0] {
        let ahead = Boundary::new(host_rate, 512).unwrap().lookahead_frames();
        let ms = 1000.0 * ahead as f64 / ENGINE_RATE;
        println!("{host_rate:>7.0} Hz: the engine runs {ahead} engine frames ({ms:.2} ms) ahead");
        assert!(
            (100..=200).contains(&ahead),
            "{host_rate} Hz: {ahead} frames is not half a {} -tap filter",
            piano_emulator_ffi::resample::SINC_LEN
        );
    }
}
