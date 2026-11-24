// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Input parity (SM161): build a throwaway collection from an offline spec,
//! run the canonical history builder exactly as the desktop does, and dump the
//! engine's input vector of every answered review and of an emulated
//! show-time query before it. Ignored; driven by ANKI_SM161_REQUEST.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use anki_proto::scheduler::RwkvHistoricalReviewFingerprintRequest;
use anki_proto::scheduler::RwkvHistoricalReviewInputsRequest;
use serde_json::json;
use serde_json::Value;

use super::*;
use crate::card::Card;
use crate::card::CardQueue;
use crate::card::CardType;
use crate::collection::CollectionBuilder;
use crate::deckconfig::DeckConfig;
use crate::deckconfig::DeckConfigId;
use crate::decks::Deck;
use crate::decks::DeckId;
use crate::decks::NativeDeckName;
use crate::notes::NoteId;
use crate::prelude::*;
use crate::revlog::RevlogEntry;
use crate::revlog::RevlogId;
use crate::revlog::RevlogReviewKind;

fn i(v: &Value) -> i64 {
    v.as_i64().unwrap()
}

fn build_collection(spec: &Value, path: &str) -> Result<()> {
    let mut col = CollectionBuilder::new(path).build()?;
    col.storage
        .set_creation_stamp(TimestampSecs(i(&spec["crt"])))?;
    let presets = i(&spec["presets"]);
    let decks = spec["decks"].as_array().unwrap();
    let cards = spec["cards"].as_array().unwrap();
    let revlog = spec["revlog"].as_array().unwrap();
    col.transact_no_undo(|col| {
        for p in 0..presets {
            let mut config = DeckConfig {
                id: DeckConfigId(1_000 + p),
                name: format!("sm161 preset {p}"),
                ..Default::default()
            };
            col.add_or_update_deck_config_legacy(&mut config)?;
        }
        for (index, deck_spec) in decks.iter().enumerate() {
            let mut deck = Deck::new_normal();
            deck.id = DeckId(2_000 + index as i64);
            deck.name = NativeDeckName::from_human_name(format!("sm161 deck {index}"));
            deck.normal_mut()?.config_id = 1_000 + i(&deck_spec[1]);
            col.storage.add_or_update_deck_with_existing_id(&deck)?;
        }
        for card_spec in cards {
            let mut card = Card::new(
                NoteId(i(&card_spec[1])),
                0,
                DeckId(2_000 + i(&card_spec[2])),
                0,
            );
            card.id = CardId(i(&card_spec[0]));
            card.ctype = CardType::Review;
            card.queue = CardQueue::Review;
            card.interval = 1;
            col.storage.add_or_update_card(&card)?;
        }
        for row in revlog {
            let entry = RevlogEntry {
                id: RevlogId(i(&row[0])),
                cid: CardId(i(&row[1])),
                usn: Usn(0),
                button_chosen: i(&row[2]) as u8,
                interval: i(&row[5]) as i32,
                last_interval: 0,
                ease_factor: i(&row[6]) as u32,
                taken_millis: i(&row[3]) as u32,
                review_kind: RevlogReviewKind::try_from(i(&row[4]) as u8).unwrap(),
            };
            col.storage.add_revlog_entry(&entry, false)?;
        }
        Ok(())
    })?;
    col.close(None)
}

/// The fixed-length day of a millisecond timestamp, as the history replay
/// computes it (rwkv_historical_day_offset).
fn day_of(millis: i64, next_day_at: i64, days_elapsed: i64) -> i64 {
    let secs = millis / 1000;
    (days_elapsed - (next_day_at - 1 - secs).max(0) / 86_400).max(0)
}

#[test]
#[ignore]
fn rwkv_sm161_input_parity_dump() {
    let spec: Value = serde_json::from_slice(
        &fs::read(std::env::var("ANKI_SM161_REQUEST").unwrap()).unwrap(),
    )
    .unwrap();
    let path = spec["collection"].as_str().unwrap().to_string();
    assert!(!Path::new(&path).exists(), "collection must be new");
    build_collection(&spec, &path).unwrap();

    let mut col = CollectionBuilder::new(&path).build().unwrap();
    let timing = col.timing_today().unwrap();
    assert_eq!(
        timing.next_day_at.0,
        i(&spec["next_day_at"]),
        "rollover moved: rebuild the spec"
    );
    // The desktop's production request: every deck config uses the card
    // creation policy (qt/aqt/rwkv_scheduler.py
    // _rwkv_review_first_review_elapsed_from_card_creation), no dynamic
    // presets, no add-on presets, deck-config first-review source.
    let configs: HashMap<i64, bool> = col
        .storage
        .all_deck_config()
        .unwrap()
        .into_iter()
        .map(|config| (config.id.0, true))
        .collect();
    let response = col
        .rwkv_historical_review_inputs(RwkvHistoricalReviewInputsRequest {
            history: Some(RwkvHistoricalReviewFingerprintRequest {
                first_review_uses_creation_by_config_id: configs,
                ..Default::default()
            }),
            first_review_uses_creation: None,
            ..Default::default()
        })
        .unwrap();

    let next_day_at = timing.next_day_at.0;
    let days_elapsed = timing.days_elapsed as i64;
    let output = PathBuf::from(spec["output"].as_str().unwrap());
    fs::create_dir_all(&output).unwrap();
    let mut features_file =
        std::io::BufWriter::new(fs::File::create(output.join("features.f32")).unwrap());
    let mut rows = Vec::new();
    let mut features = FeatureState::default();
    let mut previous_by_card: HashMap<i64, i64> = HashMap::new();
    let mut write = |kind: i64, input: &ReviewInput, review_id: i64, values: Vec<f32>| {
        assert_eq!(values.len(), CARD_FEATURES);
        for value in values {
            features_file.write_all(&value.to_le_bytes()).unwrap();
        }
        rows.push(json!([
            review_id,
            kind,
            input.card_id,
            input.note_id,
            input.deck_id,
            input.preset_id,
            input.day_offset,
            input.current_elapsed_days,
            input.current_elapsed_seconds,
            input.card_type
        ]));
    };
    for row in &response.reviews {
        // qt/aqt/rwkv_history.review_input_from_row, then the rsbridge
        // identity mapping into the engine input.
        let answer = ReviewInput {
            card_id: row.card_id,
            note_id: Some(row.note_id),
            deck_id: Some(row.deck_id),
            preset_id: Some(row.preset_id),
            is_query: false,
            ease: Some(row.ease as u8),
            duration_millis: Some(row.duration_millis),
            card_type: Some(row.card_type),
            day_offset: Some(row.day_offset),
            current_elapsed_days: Some(row.elapsed_days),
            current_elapsed_seconds: Some(row.elapsed_seconds),
            target_retentions: [None; 4],
            enforce_grade_order: false,
        };
        if let Some(&previous) = previous_by_card.get(&row.card_id) {
            // Show-time query before this answer: shown at answer - duration,
            // the card's last review at the previous kept answer.
            let shown_millis = row.review_id - row.duration_millis.max(0);
            let now = shown_millis / 1000;
            let last = previous / 1000;
            let show_day = day_of(shown_millis, next_day_at, days_elapsed);
            let previous_day = day_of(previous, next_day_at, days_elapsed);
            let next_day_after_show = next_day_at - (next_day_at - 1 - now).max(0) / 86_400 * 86_400;
            let real_seconds = (now - last).max(0);
            let mut query = answer.clone();
            query.is_query = true;
            query.ease = None;
            query.duration_millis = None;
            query.day_offset = Some(show_day);
            let learning = row.card_type == 1;
            if learning {
                // Reviewer path (_scheduling_state_elapsed): the learning
                // state's elapsed_secs; DayLearn counts whole days, the
                // intraday Learn queue the time since the answer.
                query.current_elapsed_days = None;
                query.current_elapsed_seconds = Some(if show_day > previous_day {
                    (show_day - previous_day) * 86_400
                } else {
                    real_seconds
                });
            } else {
                // Review, relearning and filtered: _elapsed_since_card_last_review.
                query.current_elapsed_days = Some((next_day_after_show - last).max(0) / 86_400);
                query.current_elapsed_seconds = Some(real_seconds);
            }
            let values = features.features_for(&query);
            write(1, &query, row.review_id, values);
            if learning {
                // Batch path (rwkv_review_input_state, CardType::Learn).
                let mut batch = query.clone();
                batch.current_elapsed_days = None;
                batch.current_elapsed_seconds = Some(real_seconds);
                let values = features.features_for(&batch);
                write(2, &batch, row.review_id, values);
            }
            // Live answer (review_answered -> rwkv_review_input with the
            // ease, at answer time): the same elapsed rules as the query, with
            // the revlog duration as the reviewer's timer.
            let answered = row.review_id / 1000;
            let answer_day = row.day_offset;
            let next_day_after_answer =
                next_day_at - (next_day_at - 1 - answered).max(0) / 86_400 * 86_400;
            let mut live = answer.clone();
            if learning {
                live.current_elapsed_days = None;
                live.current_elapsed_seconds = Some(if answer_day > previous_day {
                    (answer_day - previous_day) * 86_400
                } else {
                    (answered - last).max(0)
                });
            } else {
                live.current_elapsed_days = Some((next_day_after_answer - last).max(0) / 86_400);
                live.current_elapsed_seconds = Some((answered - last).max(0));
            }
            let values = features.features_for(&live);
            write(3, &live, row.review_id, values);
        }
        let values = features.features_for(&answer);
        write(0, &answer, row.review_id, values);
        features.store_review(&answer);
        previous_by_card.insert(row.card_id, row.review_id);
    }
    drop(write);
    features_file.flush().unwrap();
    fs::write(
        output.join("rows.json"),
        serde_json::to_vec(&json!({
            "columns": ["review_id", "kind", "card_id", "note_id", "deck_id", "preset_id",
                "day_offset", "elapsed_days", "elapsed_seconds", "card_type"],
            "rows": rows,
            "next_day_at": next_day_at,
            "days_elapsed": days_elapsed,
            "history_reviews": response.reviews.len(),
        }))
        .unwrap(),
    )
    .unwrap();
}
