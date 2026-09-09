import copy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scraping.scraper_us_open import parse_scoreboard, ScheduleUnavailable, get_us_open_matches
from us_open_predict import predict_matches, EloPredictor, TournamentEloPredictor, write_predictions_txt

NOW = datetime(2026, 9, 9, 8, tzinfo=timezone.utc)


def fixture():
    match = {"id": "123", "date": "2026-09-09T18:00Z", "timeValid": True,
             "status": {"type": {"state": "pre", "name": "STATUS_SCHEDULED", "completed": False}},
             "round": {"displayName": "Quarterfinal"},
             "competitors": [{"type": "athlete", "athlete": {"displayName": name}}
                             for name in ["Jannik Sinner", "Carlos Alcaraz"]]}
    return {"events": [{"name": "US Open", "season": {"year": 2026}, "groupings": [
        {"grouping": {"slug": "mens-singles"}, "competitions": [match]}]}]}


def competitions(data):
    return data["events"][0]["groupings"][0]["competitions"]


class ScheduleTests(unittest.TestCase):
    def test_tournament_elo_uses_only_completed_past_results(self):
        data = fixture()
        past = copy.deepcopy(competitions(data)[0])
        past.update(id="past", date="2026-09-08T18:00Z", round={"displayName": "Round 4"},
                    status={"type": {"state": "post", "name": "STATUS_FINAL", "completed": True}})
        past["competitors"][0]["winner"] = True
        past["competitors"][1]["winner"] = False
        competitions(data).append(past)
        future_result = copy.deepcopy(past)
        future_result.update(id="future", date="2026-09-10T18:00Z")
        competitions(data).append(future_result)
        matches = parse_scoreboard(data, "ATP", 2026, NOW)
        self.assertEqual(len(matches.attrs["completed_results"]), 1)
        predictor = TournamentEloPredictor(matches.attrs["completed_results"])
        predictions = predict_matches(matches, predictor, "ATP")
        self.assertGreater(predictions.iloc[0].p1_win_prob, 0.5)
        self.assertEqual(predictions.iloc[0].p1_sample_matches, 1)
        self.assertAlmostEqual(sum(predictor.elo_table.values()), sum(EloPredictor("ATP").elo_table.values()))

    def test_single_match_and_provenance(self):
        frame = parse_scoreboard(fixture(), "ATP", 2026, NOW, "fixture")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["round"], "QF")
        self.assertEqual(frame.iloc[0].source, "fixture")

    def test_exclude_other_edition_tour_and_tournament(self):
        self.assertTrue(parse_scoreboard(fixture(), "WTA", 2026, NOW).empty)
        self.assertTrue(parse_scoreboard(fixture(), "ATP", 2025, NOW).empty)
        data = fixture()
        data["events"][0]["name"] = "Australian Open"
        self.assertTrue(parse_scoreboard(data, "ATP", 2026, NOW).empty)

    def test_not_started_and_valid_date_required(self):
        for state in ["in", "post"]:
            data = fixture()
            competitions(data)[0]["status"]["type"]["state"] = state
            self.assertTrue(parse_scoreboard(data, "ATP", 2026, NOW).empty)
        for value in ["2026-09-08T18:00Z", "invalid", "2025-09-09T18:00Z", "2026-09-09T18:00"]:
            data = fixture()
            competitions(data)[0]["date"] = value
            self.assertTrue(parse_scoreboard(data, "ATP", 2026, NOW).empty)

    def test_placeholder_dedupe_and_names(self):
        data = fixture()
        competitions(data).append(copy.deepcopy(competitions(data)[0]))
        self.assertEqual(len(parse_scoreboard(data, "ATP", 2026, NOW)), 1)
        for name in ["TBD", "Qualifier A", "Winner of QF 1"]:
            data = fixture()
            competitions(data)[0]["competitors"][0]["athlete"]["displayName"] = name
            self.assertTrue(parse_scoreboard(data, "ATP", 2026, NOW).empty)
        for name in ["Danielle Collins", "Qinwen Zheng", "Iga Świątek", "Félix Auger-Aliassime"]:
            data = fixture()
            competitions(data)[0]["competitors"][0]["athlete"]["displayName"] = name
            self.assertEqual(len(parse_scoreboard(data, "ATP", 2026, NOW)), 1)

    def test_tbd_time_is_explicit(self):
        data = fixture()
        competitions(data)[0].update(timeValid=False, date="2026-09-09T04:00Z")
        result = parse_scoreboard(data, "ATP", 2026, NOW)
        self.assertEqual(len(result), 1)
        self.assertFalse(result.iloc[0].time_confirmed)

    def test_invalid_schema_and_network_failure(self):
        for data in [{}, {"events": None}]:
            with self.assertRaises(ScheduleUnavailable):
                parse_scoreboard(data, "ATP", 2026, NOW)
        import requests
        with patch("requests.Session.get", side_effect=requests.ConnectionError("offline")):
            with self.assertRaises(ScheduleUnavailable):
                get_us_open_matches("ATP", 2026)

    def test_empty_predictions_report(self):
        matches = parse_scoreboard({"events": []}, "ATP", 2026, NOW)
        preds = predict_matches(matches, EloPredictor("ATP"), "ATP")
        self.assertIn("predicted_winner", preds)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.txt"
            write_predictions_txt(preds, preds, "test", "test", path)
            self.assertIn("No confirmed upcoming", path.read_text(encoding="utf-8"))

    def test_probabilities_unknown_and_metadata(self):
        matches = parse_scoreboard(fixture(), "ATP", 2026, NOW)
        preds = predict_matches(matches, EloPredictor("ATP"), "ATP")
        self.assertAlmostEqual(preds.iloc[0].p1_win_prob + preds.iloc[0].p2_win_prob, 1)
        self.assertEqual(preds.iloc[0].scheduled_at, matches.iloc[0].scheduled_at)
        matches.loc[0, "player1"] = "Unknown Player Without Rating"
        preds = predict_matches(matches, EloPredictor("ATP"), "ATP")
        self.assertEqual(preds.iloc[0].prediction_status, "insufficient_data")
        self.assertTrue(preds.p1_win_prob.isna().all())


if __name__ == "__main__":
    unittest.main()
