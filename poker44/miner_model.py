"""Chunk-level feature extraction for Poker44 miner models."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Iterable


ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _share(count: int | float, total: int | float) -> float:
    total_f = float(total)
    return float(count) / total_f if total_f > 0 else 0.0


def _entropy(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _street_index(street: str) -> int:
    normalized = str(street or "").strip().lower()
    if normalized == "preflop":
        return 0
    if normalized == "flop":
        return 1
    if normalized == "turn":
        return 2
    if normalized in {"river", "showdown"}:
        return 3
    return 0


def _iter_actions(chunk: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for hand in chunk:
        actions = hand.get("actions") if isinstance(hand, dict) else None
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, dict):
                yield action


def extract_chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Return numeric behavior features for one miner-visible chunk group."""
    hand_count = len(chunk)
    action_counts_by_hand: list[float] = []
    player_counts: list[float] = []
    street_counts: list[float] = []
    max_street_indices: list[float] = []
    starting_stacks: list[float] = []
    pot_growth_by_hand: list[float] = []

    action_counter: Counter[str] = Counter()
    street_action_counter: Counter[str] = Counter()
    amount_values: list[float] = []
    nonzero_amount_values: list[float] = []
    pot_before_values: list[float] = []
    pot_after_values: list[float] = []
    pot_growth_values: list[float] = []
    zero_amount_count = 0

    hero_seats: list[float] = []
    hero_button_offsets: list[float] = []
    hero_button_count = 0
    hero_action_counts_by_hand: list[float] = []
    hero_last_street_indices: list[float] = []
    hero_pot_growth_by_hand: list[float] = []
    hero_action_counter: Counter[str] = Counter()
    hero_street_action_counter: Counter[str] = Counter()
    hero_street_type_counter: Counter[str] = Counter()
    hero_amount_values: list[float] = []
    hero_nonzero_amount_values: list[float] = []
    hero_amount_pot_ratios: list[float] = []
    hero_pot_before_values: list[float] = []
    hero_pot_after_values: list[float] = []
    hero_pot_growth_values: list[float] = []
    hero_call_to_pot_ratios: list[float] = []
    hero_zero_amount_count = 0
    hero_active_hand_count = 0
    hero_vpip_hand_count = 0
    hero_pfr_hand_count = 0
    hero_preflop_fold_hand_count = 0
    hero_flop_action_hand_count = 0
    hero_turn_action_hand_count = 0
    hero_river_action_hand_count = 0
    hero_facing_call_count = 0
    hero_facing_action_counter: Counter[str] = Counter()
    hero_facing_street_counter: Counter[str] = Counter()
    hero_facing_street_action_counter: Counter[str] = Counter()
    hero_starting_stacks: list[float] = []
    hero_stack_to_table_mean_values: list[float] = []
    hero_stack_rank_pct_values: list[float] = []
    hero_short_stack_count = 0
    hero_big_stack_count = 0
    hero_first_action_counter: Counter[str] = Counter()
    hero_last_action_counter: Counter[str] = Counter()
    hero_first_preflop_action_counter: Counter[str] = Counter()
    hero_last_preflop_action_counter: Counter[str] = Counter()
    hero_aggressive_actions_by_hand: list[float] = []
    hero_passive_actions_by_hand: list[float] = []
    hero_distinct_action_types_by_hand: list[float] = []
    hero_nonzero_actions_by_hand: list[float] = []
    hero_nonzero_amount_sum_by_hand: list[float] = []
    hero_street_span_by_hand: list[float] = []
    hero_same_action_transition_count = 0
    hero_action_transition_count = 0
    hero_single_action_hand_count = 0
    hero_multi_action_hand_count = 0
    hero_multi_street_hand_count = 0
    hero_preflop_call_hand_count = 0
    hero_preflop_raise_hand_count = 0
    hero_preflop_limp_hand_count = 0
    hero_preflop_open_raise_hand_count = 0
    hero_raise_hand_count = 0
    hero_bet_hand_count = 0
    hero_call_hand_count = 0
    hero_check_hand_count = 0
    hero_fold_hand_count = 0
    hero_raise_to_pot_ratios: list[float] = []

    for hand in chunk:
        if not isinstance(hand, dict):
            continue

        metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
        hero_seat = _safe_int(metadata.get("hero_seat"))
        button_seat = _safe_int(metadata.get("button_seat"))
        max_seats = _safe_int(metadata.get("max_seats")) or 0
        if hero_seat is not None:
            hero_seats.append(float(hero_seat))
            if button_seat is not None:
                if hero_seat == button_seat:
                    hero_button_count += 1
                if max_seats > 0:
                    hero_button_offsets.append(float((hero_seat - button_seat) % max_seats))

        players = hand.get("players") if isinstance(hand.get("players"), list) else []
        actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []
        streets = hand.get("streets") if isinstance(hand.get("streets"), list) else []

        player_counts.append(float(len(players)))
        action_counts_by_hand.append(float(len(actions)))
        street_counts.append(float(len(streets)))
        max_street_indices.append(
            float(max((_street_index(street.get("street", "")) for street in streets if isinstance(street, dict)), default=0))
        )

        hand_starting_stacks: list[float] = []
        hero_starting_stack: float | None = None
        for player in players:
            if isinstance(player, dict):
                starting_stack = _safe_float(player.get("starting_stack"))
                starting_stacks.append(starting_stack)
                hand_starting_stacks.append(starting_stack)
                if hero_seat is not None and _safe_int(player.get("seat")) == hero_seat:
                    hero_starting_stack = starting_stack

        if hero_starting_stack is not None:
            hero_starting_stacks.append(hero_starting_stack)
            table_mean_stack = _mean(hand_starting_stacks)
            if table_mean_stack > 0:
                hero_stack_to_table_mean_values.append(hero_starting_stack / table_mean_stack)
            if hand_starting_stacks:
                sorted_stacks = sorted(hand_starting_stacks)
                hero_stack_rank_pct_values.append(
                    _share(sum(1 for value in sorted_stacks if value <= hero_starting_stack), len(sorted_stacks))
                )
                if hero_starting_stack <= min(sorted_stacks):
                    hero_short_stack_count += 1
                if hero_starting_stack >= max(sorted_stacks):
                    hero_big_stack_count += 1

        hand_growth_values: list[float] = []
        hero_hand_growth_values: list[float] = []
        hero_hand_action_count = 0
        hero_hand_last_street_index = 0
        hero_hand_streets: set[str] = set()
        hero_hand_action_types: list[str] = []
        hero_hand_preflop_action_types: list[str] = []
        hero_hand_nonzero_amount_sum = 0.0
        hero_hand_nonzero_action_count = 0
        hero_hand_aggressive_action_count = 0
        hero_hand_passive_action_count = 0
        hero_hand_raise_count = 0
        hero_hand_bet_count = 0
        hero_hand_call_count = 0
        hero_hand_check_count = 0
        hero_hand_fold_count = 0
        hero_hand_vpip = False
        hero_hand_pfr = False
        hero_hand_preflop_call = False
        hero_hand_preflop_raise = False
        hero_hand_preflop_fold = False
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action_type") or "").strip().lower()
            if action_type:
                action_counter[action_type] += 1

            street = str(action.get("street") or "").strip().lower() or "preflop"
            street_action_counter[street] += 1

            amount = _safe_float(action.get("normalized_amount_bb"))
            pot_before = _safe_float(action.get("pot_before"))
            pot_after = _safe_float(action.get("pot_after"))
            pot_growth = max(0.0, pot_after - pot_before)

            amount_values.append(amount)
            if amount > 0:
                nonzero_amount_values.append(amount)
            else:
                zero_amount_count += 1
            pot_before_values.append(pot_before)
            pot_after_values.append(pot_after)
            pot_growth_values.append(pot_growth)
            hand_growth_values.append(pot_growth)

            actor_seat = _safe_int(action.get("actor_seat"))
            if hero_seat is None or actor_seat != hero_seat:
                continue

            hero_hand_action_count += 1
            if action_type:
                hero_action_counter[action_type] += 1
                hero_street_type_counter[f"{street}:{action_type}"] += 1
                hero_hand_action_types.append(action_type)
                if action_type in {"bet", "raise"}:
                    hero_hand_aggressive_action_count += 1
                if action_type in {"check", "call"}:
                    hero_hand_passive_action_count += 1
                if action_type == "raise":
                    hero_hand_raise_count += 1
                elif action_type == "bet":
                    hero_hand_bet_count += 1
                elif action_type == "call":
                    hero_hand_call_count += 1
                elif action_type == "check":
                    hero_hand_check_count += 1
                elif action_type == "fold":
                    hero_hand_fold_count += 1

            hero_street_action_counter[street] += 1
            hero_hand_streets.add(street)
            hero_hand_last_street_index = max(hero_hand_last_street_index, _street_index(street))

            raw_amount = _safe_float(action.get("amount"))
            hero_amount_values.append(amount)
            if amount > 0:
                hero_nonzero_amount_values.append(amount)
                hero_hand_nonzero_amount_sum += amount
                hero_hand_nonzero_action_count += 1
            else:
                hero_zero_amount_count += 1
            if raw_amount > 0 and pot_before > 0:
                hero_amount_pot_ratios.append(raw_amount / pot_before)
            raise_to = _safe_float(action.get("raise_to"))
            if raise_to > 0 and pot_before > 0:
                hero_raise_to_pot_ratios.append(raise_to / pot_before)

            hero_pot_before_values.append(pot_before)
            hero_pot_after_values.append(pot_after)
            hero_pot_growth_values.append(pot_growth)
            hero_hand_growth_values.append(pot_growth)

            call_to = _safe_float(action.get("call_to"))
            if call_to > 0:
                hero_facing_call_count += 1
                if action_type:
                    hero_facing_action_counter[action_type] += 1
                    hero_facing_street_action_counter[f"{street}:{action_type}"] += 1
                hero_facing_street_counter[street] += 1
                if pot_before > 0:
                    hero_call_to_pot_ratios.append(call_to / pot_before)

            if street == "preflop":
                if action_type:
                    hero_hand_preflop_action_types.append(action_type)
                if action_type in {"call", "bet", "raise"} and amount > 0:
                    hero_hand_vpip = True
                if action_type in {"bet", "raise"}:
                    hero_hand_pfr = True
                if action_type == "call":
                    hero_hand_preflop_call = True
                if action_type in {"bet", "raise"}:
                    hero_hand_preflop_raise = True
                if action_type == "fold":
                    hero_hand_preflop_fold = True

        pot_growth_by_hand.append(sum(hand_growth_values))
        hero_action_counts_by_hand.append(float(hero_hand_action_count))
        hero_last_street_indices.append(float(hero_hand_last_street_index))
        hero_pot_growth_by_hand.append(sum(hero_hand_growth_values))
        hero_aggressive_actions_by_hand.append(float(hero_hand_aggressive_action_count))
        hero_passive_actions_by_hand.append(float(hero_hand_passive_action_count))
        hero_distinct_action_types_by_hand.append(float(len(set(hero_hand_action_types))))
        hero_nonzero_actions_by_hand.append(float(hero_hand_nonzero_action_count))
        hero_nonzero_amount_sum_by_hand.append(hero_hand_nonzero_amount_sum)
        hero_street_span_by_hand.append(float(len(hero_hand_streets)))

        if hero_hand_action_types:
            hero_first_action_counter[hero_hand_action_types[0]] += 1
            hero_last_action_counter[hero_hand_action_types[-1]] += 1
            for previous_action, next_action in zip(hero_hand_action_types, hero_hand_action_types[1:]):
                hero_action_transition_count += 1
                if previous_action == next_action:
                    hero_same_action_transition_count += 1
        if hero_hand_preflop_action_types:
            hero_first_preflop_action_counter[hero_hand_preflop_action_types[0]] += 1
            hero_last_preflop_action_counter[hero_hand_preflop_action_types[-1]] += 1

        if hero_hand_action_count > 0:
            hero_active_hand_count += 1
        if hero_hand_action_count == 1:
            hero_single_action_hand_count += 1
        if hero_hand_action_count >= 2:
            hero_multi_action_hand_count += 1
        if len(hero_hand_streets) >= 2:
            hero_multi_street_hand_count += 1
        if hero_hand_vpip:
            hero_vpip_hand_count += 1
        if hero_hand_pfr:
            hero_pfr_hand_count += 1
        if hero_hand_preflop_call:
            hero_preflop_call_hand_count += 1
        if hero_hand_preflop_raise:
            hero_preflop_raise_hand_count += 1
        if hero_hand_preflop_call and not hero_hand_preflop_raise:
            hero_preflop_limp_hand_count += 1
        if hero_hand_preflop_action_types and hero_hand_preflop_action_types[0] in {"bet", "raise"}:
            hero_preflop_open_raise_hand_count += 1
        if hero_hand_preflop_fold:
            hero_preflop_fold_hand_count += 1
        if hero_hand_raise_count > 0:
            hero_raise_hand_count += 1
        if hero_hand_bet_count > 0:
            hero_bet_hand_count += 1
        if hero_hand_call_count > 0:
            hero_call_hand_count += 1
        if hero_hand_check_count > 0:
            hero_check_hand_count += 1
        if hero_hand_fold_count > 0:
            hero_fold_hand_count += 1
        if "flop" in hero_hand_streets:
            hero_flop_action_hand_count += 1
        if "turn" in hero_hand_streets:
            hero_turn_action_hand_count += 1
        if "river" in hero_hand_streets:
            hero_river_action_hand_count += 1

    total_actions = sum(action_counter.values())
    total_street_actions = sum(street_action_counter.values())
    aggression_count = action_counter["bet"] + action_counter["raise"]
    passive_count = action_counter["check"] + action_counter["call"]
    hero_total_actions = sum(hero_action_counter.values())
    hero_total_street_actions = sum(hero_street_action_counter.values())
    hero_aggression_count = hero_action_counter["bet"] + hero_action_counter["raise"]
    hero_passive_count = hero_action_counter["check"] + hero_action_counter["call"]
    rounded_hero_amounts = [round(value, 2) for value in hero_nonzero_amount_values]
    hero_repeated_amount_share = (
        _share(Counter(rounded_hero_amounts).most_common(1)[0][1], len(rounded_hero_amounts))
        if rounded_hero_amounts
        else 0.0
    )

    features: dict[str, float] = {
        "hand_count": float(hand_count),
        "total_actions": float(total_actions),
        "mean_action_count": _mean(action_counts_by_hand),
        "std_action_count": _std(action_counts_by_hand),
        "median_action_count": _median(action_counts_by_hand),
        "mean_player_count": _mean(player_counts),
        "std_player_count": _std(player_counts),
        "min_player_count": min(player_counts) if player_counts else 0.0,
        "max_player_count": max(player_counts) if player_counts else 0.0,
        "mean_street_count": _mean(street_counts),
        "std_street_count": _std(street_counts),
        "mean_max_street_index": _mean(max_street_indices),
        "river_or_showdown_hand_share": _share(sum(1 for value in max_street_indices if value >= 3), hand_count),
        "flop_plus_hand_share": _share(sum(1 for value in max_street_indices if value >= 1), hand_count),
        "mean_starting_stack": _mean(starting_stacks),
        "std_starting_stack": _std(starting_stacks),
        "mean_normalized_amount_bb": _mean(amount_values),
        "std_normalized_amount_bb": _std(amount_values),
        "median_normalized_amount_bb": _median(amount_values),
        "mean_nonzero_amount_bb": _mean(nonzero_amount_values),
        "std_nonzero_amount_bb": _std(nonzero_amount_values),
        "mean_pot_before": _mean(pot_before_values),
        "mean_pot_after": _mean(pot_after_values),
        "mean_pot_growth": _mean(pot_growth_values),
        "std_pot_growth": _std(pot_growth_values),
        "mean_hand_pot_growth": _mean(pot_growth_by_hand),
        "std_hand_pot_growth": _std(pot_growth_by_hand),
        "zero_amount_share": _share(zero_amount_count, len(amount_values)),
        "aggression_rate": _share(aggression_count, total_actions),
        "passive_rate": _share(passive_count, total_actions),
        "raise_call_ratio": float(action_counter["raise"]) / max(float(action_counter["call"]), 1.0),
        "bet_check_ratio": float(action_counter["bet"]) / max(float(action_counter["check"]), 1.0),
        "action_entropy": _entropy(action_counter, total_actions),
        "hero_seat_known_share": _share(len(hero_seats), hand_count),
        "hero_seat_mean": _mean(hero_seats),
        "hero_seat_std": _std(hero_seats),
        "hero_button_share": _share(hero_button_count, len(hero_seats)),
        "hero_button_offset_mean": _mean(hero_button_offsets),
        "hero_button_offset_std": _std(hero_button_offsets),
        "hero_mean_starting_stack": _mean(hero_starting_stacks),
        "hero_std_starting_stack": _std(hero_starting_stacks),
        "hero_stack_to_table_mean": _mean(hero_stack_to_table_mean_values),
        "hero_stack_to_table_mean_std": _std(hero_stack_to_table_mean_values),
        "hero_stack_rank_pct_mean": _mean(hero_stack_rank_pct_values),
        "hero_stack_rank_pct_std": _std(hero_stack_rank_pct_values),
        "hero_short_stack_share": _share(hero_short_stack_count, len(hero_starting_stacks)),
        "hero_big_stack_share": _share(hero_big_stack_count, len(hero_starting_stacks)),
        "hero_total_actions": float(hero_total_actions),
        "hero_action_share": _share(hero_total_actions, total_actions),
        "hero_actions_per_hand": _share(hero_total_actions, hand_count),
        "hero_active_hand_share": _share(hero_active_hand_count, hand_count),
        "hero_single_action_hand_share": _share(hero_single_action_hand_count, hand_count),
        "hero_multi_action_hand_share": _share(hero_multi_action_hand_count, hand_count),
        "hero_multi_street_hand_share": _share(hero_multi_street_hand_count, hand_count),
        "hero_mean_action_count": _mean(hero_action_counts_by_hand),
        "hero_std_action_count": _std(hero_action_counts_by_hand),
        "hero_median_action_count": _median(hero_action_counts_by_hand),
        "hero_mean_aggressive_actions_per_hand": _mean(hero_aggressive_actions_by_hand),
        "hero_mean_passive_actions_per_hand": _mean(hero_passive_actions_by_hand),
        "hero_mean_distinct_action_types_per_hand": _mean(hero_distinct_action_types_by_hand),
        "hero_mean_nonzero_actions_per_hand": _mean(hero_nonzero_actions_by_hand),
        "hero_mean_nonzero_amount_sum_per_hand": _mean(hero_nonzero_amount_sum_by_hand),
        "hero_std_nonzero_amount_sum_per_hand": _std(hero_nonzero_amount_sum_by_hand),
        "hero_mean_street_span_per_hand": _mean(hero_street_span_by_hand),
        "hero_same_action_transition_share": _share(
            hero_same_action_transition_count, hero_action_transition_count
        ),
        "hero_mean_last_street_index": _mean(hero_last_street_indices),
        "hero_river_or_showdown_hand_share": _share(sum(1 for value in hero_last_street_indices if value >= 3), hand_count),
        "hero_flop_action_hand_share": _share(hero_flop_action_hand_count, hand_count),
        "hero_turn_action_hand_share": _share(hero_turn_action_hand_count, hand_count),
        "hero_river_action_hand_share": _share(hero_river_action_hand_count, hand_count),
        "hero_vpip_hand_share": _share(hero_vpip_hand_count, hand_count),
        "hero_pfr_hand_share": _share(hero_pfr_hand_count, hand_count),
        "hero_pfr_vpip_ratio": float(hero_pfr_hand_count) / max(float(hero_vpip_hand_count), 1.0),
        "hero_preflop_call_hand_share": _share(hero_preflop_call_hand_count, hand_count),
        "hero_preflop_raise_hand_share": _share(hero_preflop_raise_hand_count, hand_count),
        "hero_preflop_limp_hand_share": _share(hero_preflop_limp_hand_count, hand_count),
        "hero_preflop_limp_vpip_ratio": float(hero_preflop_limp_hand_count) / max(float(hero_vpip_hand_count), 1.0),
        "hero_preflop_open_raise_hand_share": _share(hero_preflop_open_raise_hand_count, hand_count),
        "hero_preflop_fold_hand_share": _share(hero_preflop_fold_hand_count, hand_count),
        "hero_raise_hand_share": _share(hero_raise_hand_count, hand_count),
        "hero_bet_hand_share": _share(hero_bet_hand_count, hand_count),
        "hero_call_hand_share": _share(hero_call_hand_count, hand_count),
        "hero_check_hand_share": _share(hero_check_hand_count, hand_count),
        "hero_fold_hand_share": _share(hero_fold_hand_count, hand_count),
        "hero_mean_normalized_amount_bb": _mean(hero_amount_values),
        "hero_std_normalized_amount_bb": _std(hero_amount_values),
        "hero_median_normalized_amount_bb": _median(hero_amount_values),
        "hero_mean_nonzero_amount_bb": _mean(hero_nonzero_amount_values),
        "hero_std_nonzero_amount_bb": _std(hero_nonzero_amount_values),
        "hero_repeated_nonzero_amount_share": hero_repeated_amount_share,
        "hero_unique_nonzero_amount_ratio": _share(len(set(rounded_hero_amounts)), len(rounded_hero_amounts)),
        "hero_mean_amount_pot_ratio": _mean(hero_amount_pot_ratios),
        "hero_std_amount_pot_ratio": _std(hero_amount_pot_ratios),
        "hero_median_amount_pot_ratio": _median(hero_amount_pot_ratios),
        "hero_mean_raise_to_pot_ratio": _mean(hero_raise_to_pot_ratios),
        "hero_std_raise_to_pot_ratio": _std(hero_raise_to_pot_ratios),
        "hero_median_raise_to_pot_ratio": _median(hero_raise_to_pot_ratios),
        "hero_mean_pot_before": _mean(hero_pot_before_values),
        "hero_mean_pot_after": _mean(hero_pot_after_values),
        "hero_mean_pot_growth": _mean(hero_pot_growth_values),
        "hero_std_pot_growth": _std(hero_pot_growth_values),
        "hero_mean_hand_pot_growth": _mean(hero_pot_growth_by_hand),
        "hero_std_hand_pot_growth": _std(hero_pot_growth_by_hand),
        "hero_zero_amount_share": _share(hero_zero_amount_count, len(hero_amount_values)),
        "hero_aggression_rate": _share(hero_aggression_count, hero_total_actions),
        "hero_passive_rate": _share(hero_passive_count, hero_total_actions),
        "hero_aggression_lift": _share(hero_aggression_count, hero_total_actions) - _share(aggression_count, total_actions),
        "hero_passive_lift": _share(hero_passive_count, hero_total_actions) - _share(passive_count, total_actions),
        "hero_raise_call_ratio": float(hero_action_counter["raise"]) / max(float(hero_action_counter["call"]), 1.0),
        "hero_bet_check_ratio": float(hero_action_counter["bet"]) / max(float(hero_action_counter["check"]), 1.0),
        "hero_action_entropy": _entropy(hero_action_counter, hero_total_actions),
        "hero_facing_call_share": _share(hero_facing_call_count, hero_total_actions),
        "hero_mean_call_to_pot_ratio": _mean(hero_call_to_pot_ratios),
        "hero_fold_when_facing_rate": _share(hero_facing_action_counter["fold"], hero_facing_call_count),
        "hero_call_when_facing_rate": _share(hero_facing_action_counter["call"], hero_facing_call_count),
        "hero_raise_when_facing_rate": _share(hero_facing_action_counter["raise"], hero_facing_call_count),
    }

    for action_type in ACTION_TYPES:
        features[f"{action_type}_rate"] = _share(action_counter[action_type], total_actions)
        features[f"{action_type}_per_hand"] = _share(action_counter[action_type], hand_count)
        features[f"hero_{action_type}_rate"] = _share(hero_action_counter[action_type], hero_total_actions)
        features[f"hero_{action_type}_per_hand"] = _share(hero_action_counter[action_type], hand_count)
        features[f"hero_first_{action_type}_share"] = _share(
            hero_first_action_counter[action_type], hero_active_hand_count
        )
        features[f"hero_last_{action_type}_share"] = _share(
            hero_last_action_counter[action_type], hero_active_hand_count
        )
        features[f"hero_first_preflop_{action_type}_share"] = _share(
            hero_first_preflop_action_counter[action_type],
            sum(hero_first_preflop_action_counter.values()),
        )
        features[f"hero_last_preflop_{action_type}_share"] = _share(
            hero_last_preflop_action_counter[action_type],
            sum(hero_last_preflop_action_counter.values()),
        )

    for street in STREETS:
        features[f"{street}_action_share"] = _share(street_action_counter[street], total_street_actions)
        features[f"hero_{street}_action_share"] = _share(
            hero_street_action_counter[street], hero_total_street_actions
        )
        for action_type in ACTION_TYPES:
            street_actions = hero_street_action_counter[street]
            features[f"hero_{street}_{action_type}_rate"] = _share(
                hero_street_type_counter[f"{street}:{action_type}"], street_actions
            )
            facing_street_actions = hero_facing_street_counter[street]
            features[f"hero_{street}_{action_type}_when_facing_rate"] = _share(
                hero_facing_street_action_counter[f"{street}:{action_type}"], facing_street_actions
            )

    return features
