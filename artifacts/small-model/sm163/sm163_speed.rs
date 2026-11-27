// SM163 clean speed re-measure (temporary, untracked example; never committed
// to the fork). Synthetic data only.
//
// One process = one runtime x one state precision:
// 1. builds a large synthetic history (new cards, Learning steps, some
//    crossing days, reviews, lapses with Relearning, a few very long cards),
// 2. `rebuild`: the first-launch state build as the app runs it: runtime
//    load, state-only replay in 16,384-review chunks, a full recovery
//    checkpoint 8 days before the last review and a final delta checkpoint
//    into a temporary state store,
// 3. `latency`: per-review show-time prediction and answer on the resident
//    state (state lookup + `RwkvRuntime::review`, as the desktop calls it),
//    for Review cards and Learning / Relearning cards, p50 / p95,
// 4. `sweep`: R-only queue scoring of every reviewed card from the resident
//    state in batches of 32 to 1,024.
//
// Env: SM163_RUNTIME (big-pair, small-pair, tiny-pair, big, small, tiny),
// SM163_PRECISION, SM163_CARDS (14000), SM163_DAYS (730), SM163_OUT (json
// lines), SM163_WORK (scratch folder for the checkpoint store), SM163_TAG.

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anki::rwkv::ModelKind;
use anki::rwkv::ReviewInput;
use anki::rwkv::ReviewState;
use anki::rwkv::RwkvRuntime;
use anki::rwkv::StatePrecision;

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    New,
    Learn,
    Review,
    Relearn,
}
impl Kind {
    fn code(self) -> i64 {
        match self {
            Kind::New => 0,
            Kind::Learn => 1,
            Kind::Review => 2,
            Kind::Relearn => 3,
        }
    }
}

struct Event {
    t: i64,
    card: i64,
    ease: u8,
    kind: Kind,
}

struct CardEnd {
    last_t: i64,
    kind: Kind,
    count: u32,
}

const BASE_DAY: i64 = 1_000;
const DAY: i64 = 86_400;

fn note_of(card: i64) -> i64 {
    card / 2
}

fn history(cards: i64, days: i64) -> (Vec<Event>, HashMap<i64, CardEnd>) {
    let end = days * DAY;
    let mut rng = Rng(0x5163_2026_1125);
    let mut events = Vec::new();
    let mut ends = HashMap::new();
    let learn_tail = (cards / 35).max(50);
    for card in 0..cards {
        // The last cards are introduced on the last day and stay in Learning.
        let intro_day = if card >= cards - learn_tail {
            days - 1
        } else {
            (rng.f().powf(0.8) * (days - 2) as f64) as i64
        };
        let leech = rng.f() < 0.012;
        let growth = if leech {
            1.08
        } else {
            1.25 + 1.75 * rng.f()
        };
        let lapse = if leech { 0.3 } else { 0.04 + 0.12 * rng.f() };
        let mut t = intro_day * DAY + 8 * 3_600 + (rng.f() * 12.0 * 3_600.0) as i64;
        let mut kind = Kind::New;
        let mut ivl = 1.0_f64;
        let mut count = 0u32;
        let mut last_t = t;
        let tail = card >= cards - learn_tail;
        while t < end && count < 600 {
            let r = rng.f();
            let ease = match kind {
                Kind::New => {
                    if r < 0.3 {
                        1
                    } else if r < 0.9 {
                        3
                    } else {
                        4
                    }
                }
                Kind::Learn | Kind::Relearn => {
                    if r < 0.2 {
                        1
                    } else {
                        3
                    }
                }
                Kind::Review => {
                    if r < lapse {
                        1
                    } else if r < lapse + 0.1 {
                        2
                    } else if r < 0.92 {
                        3
                    } else {
                        4
                    }
                }
            };
            events.push(Event {
                t,
                card,
                ease,
                kind,
            });
            count += 1;
            last_t = t;
            let (next_kind, wait) = match (kind, ease) {
                (Kind::New, 4) => {
                    ivl = 4.0;
                    (Kind::Review, 4 * DAY)
                }
                (Kind::New, 1) => (Kind::Learn, 60),
                (Kind::New, _) => (Kind::Learn, 600),
                (Kind::Learn, 1) => (Kind::Learn, 60),
                (Kind::Learn, _) => {
                    // A tenth of the cards keep a one-day Learning step.
                    if rng.f() < 0.1 && count < 4 {
                        (Kind::Learn, DAY)
                    } else {
                        ivl = 1.0;
                        (Kind::Review, DAY)
                    }
                }
                (Kind::Relearn, 1) => (Kind::Relearn, 600),
                (Kind::Relearn, _) => (Kind::Review, (ivl.round() as i64).max(1) * DAY),
                (Kind::Review, 1) => {
                    ivl = (ivl * 0.3).max(1.0);
                    (Kind::Relearn, 600)
                }
                (Kind::Review, 2) => {
                    ivl = (ivl * 1.2).max(ivl + 1.0);
                    (Kind::Review, ivl as i64 * DAY)
                }
                (Kind::Review, 3) => {
                    ivl = (ivl * growth).max(ivl + 1.0);
                    (Kind::Review, ivl as i64 * DAY)
                }
                (Kind::Review, _) => {
                    ivl = (ivl * growth * 1.3).max(ivl + 1.0);
                    (Kind::Review, ivl as i64 * DAY)
                }
            };
            kind = next_kind;
            // Day-long waits land at a random time of the target day.
            t += if wait >= DAY {
                wait - (t % DAY) + 7 * 3_600 + (rng.f() * 14.0 * 3_600.0) as i64
            } else {
                wait
            };
            if tail && count >= 1 + (card % 3) as u32 {
                break;
            }
        }
        if count > 0 {
            ends.insert(
                card,
                CardEnd {
                    last_t,
                    kind,
                    count,
                },
            );
        }
    }
    events.sort_by_key(|e| (e.t, e.card));
    (events, ends)
}

fn inputs(events: &[Event]) -> Vec<ReviewInput> {
    let mut last: HashMap<i64, i64> = HashMap::new();
    let mut rng = Rng(77);
    events
        .iter()
        .map(|e| {
            let (days, seconds) = match last.insert(e.card, e.t) {
                Some(previous) => (e.t / DAY - previous / DAY, e.t - previous),
                None => (0, 0),
            };
            ReviewInput {
                card_id: 10_000 + e.card,
                note_id: Some(5_000 + note_of(e.card)),
                deck_id: Some(1 + e.card % 3),
                preset_id: Some(1),
                is_query: false,
                ease: Some(e.ease),
                duration_millis: Some(1_000 + (rng.next() % 12_000) as i64),
                card_type: Some(e.kind.code()),
                day_offset: Some(BASE_DAY + e.t / DAY),
                current_elapsed_days: Some(days),
                current_elapsed_seconds: Some(seconds),
                target_retentions: [None; 4],
                enforce_grade_order: true,
            }
        })
        .collect()
}

fn weights(folder: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../qt/aqt/rwkv_inference");
    if folder == "legacy" {
        root.join("RWKV_trained_on_5000_10000.bin")
    } else {
        root.join(folder).join("weights.f32.bin")
    }
}

fn load(runtime: &str, precision: StatePrecision) -> RwkvRuntime {
    let (main, instant) = match runtime {
        "big-pair" => ("legacy", Some("sm99")),
        "small-pair" => ("sm118", Some("sm113")),
        "tiny-pair" => ("sm130th", Some("sm130io")),
        "big" => ("legacy", None),
        "small" => ("sm118", None),
        "tiny" => ("sm130th", None),
        other => panic!("unknown runtime {other}"),
    };
    let kind = |k: &str| ModelKind::parse(k).unwrap();
    match instant {
        None => RwkvRuntime::load_with_precision(weights(main), 0.9, 36_500, precision, kind(main)),
        Some(i) => RwkvRuntime::load_paired(
            weights(main),
            weights(i),
            0.9,
            36_500,
            precision,
            kind(main),
            kind(i),
        ),
    }
    .unwrap()
}

fn int_map(values: &HashMap<i64, i64>) -> Vec<u8> {
    let mut items: Vec<_> = values.iter().map(|(k, v)| (*k, *v)).collect();
    items.sort_unstable();
    let mut out = Vec::with_capacity(4 + items.len() * 16);
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for (key, value) in items {
        out.extend_from_slice(&key.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn pct(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 - 1.0) * p).round() as usize;
    values[index]
}

fn main() {
    let runtime_name = env("SM163_RUNTIME", "small-pair");
    let precision_name = env("SM163_PRECISION", "compact_cards_int8");
    let cards: i64 = env("SM163_CARDS", "14000").parse().unwrap();
    let days: i64 = env("SM163_DAYS", "730").parse().unwrap();
    let tag = env("SM163_TAG", "");
    let work = PathBuf::from(env("SM163_WORK", "."));
    fs::create_dir_all(&work).unwrap();
    let ops = env("SM163_OPS", "rebuild,latency,sweep");
    let precision = match StatePrecision::parse(&precision_name) {
        Ok(p) => p,
        Err(_) => {
            println!("skip: precision {precision_name} unknown to this build");
            return;
        }
    };

    let started = Instant::now();
    let (events, ends) = history(cards, days);
    let reviews = inputs(&events);
    let history_ms = ms(started);
    let max_count = ends.values().map(|e| e.count).max().unwrap_or(0);
    let long_cards = ends.values().filter(|e| e.count >= 128).count();
    let learning_now = ends
        .values()
        .filter(|e| matches!(e.kind, Kind::Learn | Kind::Relearn))
        .count();

    // Rebuild.
    let phase = Instant::now();
    let mut runtime = load(&runtime_name, precision);
    let load_ms = ms(phase);
    let last_t = events.last().unwrap().t;
    let checkpoint_at = events.partition_point(|e| e.t < last_t - 8 * DAY);
    let endpoints = [checkpoint_at, reviews.len()];
    let store = work.join(format!("store-{}.sqlite3", std::process::id()));
    for suffix in ["", "-wal", "-shm", "-journal"] {
        let _ = fs::remove_file(format!("{}{suffix}", store.display()));
    }
    let generation = format!("{:016x}{:016x}", Rng(last_t as u64).next(), std::process::id());
    let maps = |upto: usize| {
        let mut previous_id = HashMap::new();
        let mut previous_ivl = HashMap::new();
        let mut count = HashMap::new();
        let mut last_seen: HashMap<i64, i64> = HashMap::new();
        for e in &events[..upto] {
            let id = 10_000 + e.card;
            previous_id.insert(id, 1_600_000_000_000 + e.t * 1000);
            if let Some(prev) = last_seen.insert(id, e.t) {
                previous_ivl.insert(id, (e.t - prev) / DAY);
            }
            *count.entry(id).or_insert(0) += 1;
        }
        (int_map(&previous_id), int_map(&previous_ivl), int_map(&count))
    };
    let mut replay_ms = 0.0;
    let mut checkpoint_ms = 0.0;
    let mut head = None;
    let mut processed = 0;
    let mut endpoint = 0;
    let rebuild_started = Instant::now();
    while processed < reviews.len() {
        let end_index = (processed + 16_384).min(endpoints[endpoint]);
        let phase = Instant::now();
        runtime
            .warm_up_reviews(reviews[processed..end_index].to_vec(), false)
            .unwrap();
        replay_ms += ms(phase);
        processed = end_index;
        if processed == endpoints[endpoint] {
            let phase = Instant::now();
            let full = endpoint == 0;
            let (a, b, c) = maps(processed);
            head = Some(
                runtime
                    .write_warm_up_state_checkpoint(
                        store.clone(),
                        &generation,
                        if full { None } else { head },
                        1_600_000_000_000 + events[processed - 1].t * 1000,
                        processed as i64,
                        "sm163-synthetic",
                        "sm163",
                        &a,
                        &b,
                        &c,
                        full,
                        false,
                    )
                    .unwrap(),
            );
            checkpoint_ms += ms(phase);
            endpoint += 1;
        }
    }
    let phase = Instant::now();
    runtime.finish_warm_up_state_checkpoints().unwrap();
    checkpoint_ms += ms(phase);
    let rebuild_ms = ms(rebuild_started) + load_ms;
    let store_bytes = fs::metadata(&store).map_or(0, |m| m.len());
    for suffix in ["", "-wal", "-shm", "-journal"] {
        let _ = fs::remove_file(format!("{}{suffix}", store.display()));
    }

    let mut out = String::new();
    out += &format!(
        "{{\"tag\":\"{tag}\",\"runtime\":\"{runtime_name}\",\"precision\":\"{precision_name}\",\
         \"threads\":\"{}\",\"reviews\":{},\"cards\":{},\"max_answers\":{max_count},\
         \"cards_128_plus\":{long_cards},\"learning_now\":{learning_now},\
         \"history_ms\":{history_ms:.1},\"rebuild\":{{\"load_ms\":{load_ms:.1},\
         \"replay_ms\":{replay_ms:.1},\"checkpoint_ms\":{checkpoint_ms:.1},\
         \"total_ms\":{rebuild_ms:.1},\"replay_reviews_per_s\":{:.0},\
         \"total_reviews_per_s\":{:.0},\"store_bytes\":{store_bytes}}}",
        env("RAYON_NUM_THREADS", "default"),
        reviews.len(),
        ends.len(),
        reviews.len() as f64 / (replay_ms / 1000.0),
        reviews.len() as f64 / (rebuild_ms / 1000.0),
    );

    let now = days * DAY + 10 * 3_600;
    let query = |card: i64, end: &CardEnd, at: i64, answer: Option<u8>| ReviewInput {
        card_id: 10_000 + card,
        note_id: Some(5_000 + note_of(card)),
        deck_id: Some(1 + card % 3),
        preset_id: Some(1),
        is_query: answer.is_none(),
        ease: answer,
        duration_millis: answer.map(|_| 4_000),
        card_type: Some(end.kind.code()),
        day_offset: Some(BASE_DAY + at / DAY),
        current_elapsed_days: Some(at / DAY - end.last_t / DAY),
        current_elapsed_seconds: Some(at - end.last_t),
        target_retentions: [Some(0.9); 4],
        enforce_grade_order: true,
    };

    if ops.contains("latency") {
        let mut ids: Vec<i64> = ends.keys().copied().collect();
        ids.sort_unstable();
        let mut rng = Rng(163);
        let mut review_cards = Vec::new();
        let mut learning_cards = Vec::new();
        for id in &ids {
            let e = &ends[id];
            match e.kind {
                Kind::Review if rng.f() < 0.05 => review_cards.push(*id),
                Kind::Learn | Kind::Relearn => learning_cards.push(*id),
                _ => {}
            }
        }
        review_cards.truncate(400);
        learning_cards.truncate(400);
        let repeats = 5;
        let mut json = Vec::new();
        for (label, list) in [("review", &review_cards), ("learning", &learning_cards)] {
            let mut show = Vec::new();
            let mut answer = Vec::new();
            for _ in 0..repeats {
                for card in list.iter() {
                    let end = &ends[card];
                    // Learning cards are shown when their step is due (or
                    // on the next day for a one-day step); reviews today.
                    let at = if end.kind == Kind::Review {
                        now
                    } else {
                        (end.last_t + 600).max(last_t + 60)
                    };
                    let q = query(*card, end, at, None);
                    let phase = Instant::now();
                    let state = runtime.warm_up_state(&q).unwrap();
                    let output = runtime
                        .review(
                            q,
                            ReviewState {
                                card: state.card.as_deref(),
                                deck: state.deck.as_deref(),
                                note: state.note.as_deref(),
                                preset: state.preset.as_deref(),
                                global: state.global.as_deref(),
                            },
                        )
                        .unwrap();
                    show.push(phase.elapsed().as_secs_f64() * 1e6);
                    std::hint::black_box(output.retrievability);
                    let a = query(*card, end, at + 8, Some(3));
                    let phase = Instant::now();
                    let state = runtime.warm_up_state(&a).unwrap();
                    let output = runtime
                        .review(
                            a,
                            ReviewState {
                                card: state.card.as_deref(),
                                deck: state.deck.as_deref(),
                                note: state.note.as_deref(),
                                preset: state.preset.as_deref(),
                                global: state.global.as_deref(),
                            },
                        )
                        .unwrap();
                    answer.push(phase.elapsed().as_secs_f64() * 1e6);
                    std::hint::black_box(output.card_state.len());
                }
            }
            json.push(format!(
                "\"{label}\":{{\"cards\":{},\"calls\":{},\"show_p50_us\":{:.1},\"show_p95_us\":{:.1},\
                 \"answer_p50_us\":{:.1},\"answer_p95_us\":{:.1}}}",
                list.len(),
                show.len(),
                pct(&mut show, 0.5),
                pct(&mut show, 0.95),
                pct(&mut answer, 0.5),
                pct(&mut answer, 0.95),
            ));
        }
        out += &format!(",\"latency\":{{{}}}", json.join(","));
    }

    if ops.contains("sweep") {
        let mut ids: Vec<i64> = ends.keys().copied().collect();
        ids.sort_unstable();
        let queries: Vec<ReviewInput> = ids
            .iter()
            .map(|card| query(*card, &ends[card], now, None))
            .collect();
        let mut json = Vec::new();
        for batch in [32usize, 64, 128, 256, 512, 1024] {
            let mut passes = Vec::new();
            for _ in 0..3 {
                let phase = Instant::now();
                let mut sum = 0.0f64;
                for chunk in queries.chunks(batch) {
                    let r = runtime
                        .predict_retrievability_many_from_warm_up(chunk.to_vec())
                        .unwrap();
                    sum += r.iter().map(|&x| f64::from(x)).sum::<f64>();
                }
                passes.push(ms(phase));
                std::hint::black_box(sum);
            }
            let median_ms = pct(&mut passes, 0.5);
            json.push(format!(
                "\"{batch}\":{{\"ms\":{median_ms:.1},\"queries_per_s\":{:.0},\"min_ms\":{:.1},\"max_ms\":{:.1}}}",
                queries.len() as f64 / (median_ms / 1000.0),
                passes[0],
                passes[2],
            ));
        }
        out += &format!(",\"sweep_queries\":{},\"sweep\":{{{}}}", queries.len(), json.join(","));
    }
    out += "}";
    println!("{out}");
    if let Ok(path) = std::env::var("SM163_OUT") {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(file, "{out}").unwrap();
    }
}
