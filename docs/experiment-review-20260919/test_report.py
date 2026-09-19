"""Offline checks for the exact published DEV snapshot; no GPU or network needed."""
import csv
import hashlib
import io
import json
import math
import re
import statistics
import unittest

import make_report as report


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(report.read_input('readouts.json'))
        cls.files = report.build()

    def test_complete_snapshot_and_unique_identities(self):
        self.assertEqual(self.data['errors'], [])
        self.assertEqual(len(self.data['volume']), 75)
        self.assertEqual(len(self.data['failures']), 6)
        self.assertEqual(len(self.data['fixed']), 99)
        for name in ('volume', 'fixed'):
            rows = self.data[name]
            keys = [(r['model'], r['arm'], r['seed']) for r in rows]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual({r['seed'] for r in rows}, {71, 72, 73})
            for r in rows:
                for field in ('checkpoint_sha256', 'spec_sha256', 'source_sha256'):
                    self.assertRegex(r[field], r'^[a-f0-9]{64}$')
                self.assertEqual(r['step'], 31 if name == 'volume' else 18)

    def test_uniform_has_matching_budget(self):
        selections = {s['model']: s['recipe'] for s in self.data['selections']}
        for r in self.data['volume']:
            if r['arm'] != 'uniform_matched_q':
                continue
            self.assertAlmostEqual(report.clean_share(r), selections[r['model']]['Clean'])
            self.assertEqual(len({r['shares'][g] for g in report.GROUPS[1:]}), 1)
        mismatched = {r['model'] for r in self.data['fixed']
                      if r['arm'] == 'uniform_reference'
                      and abs(report.clean_share(r) - selections[r['model']]['Clean']) > 1e-8}
        self.assertEqual(mismatched, {'google/gemma-4-12B-it', 'mistralai/Mistral-7B-Instruct-v0.3'})

    def test_fixture_and_checkpoint_binding(self):
        for rows in (self.data['volume'] + self.data['fixed'], list(self.data['knowledge'].values())):
            for field in ('generation_fixture', 'scorer'):
                self.assertEqual(len({r[field] for r in rows}), 1)
                self.assertRegex(rows[0][field], r'^[a-f0-9]{64}$')
        for r in self.data['fixed']:
            k = self.data['knowledge'][r['checkpoint_sha256']]
            self.assertEqual((r['model'], r['seed']), (k['model'], k['seed']))
            self.assertEqual(k['checkpoint_sha256'], r['checkpoint_sha256'])
            self.assertNotIn('PARA', k['nll'])

    def test_all_primary_measurements_finite_and_bmk_bounded(self):
        for r in csv.DictReader(io.StringIO(self.files['data/measurements_long.csv'])):
            value = float(r['value'])
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0)
            if r['metric'] == 'gen':
                self.assertLessEqual(value, 1)

    def test_missing_seeds_not_imputed_or_counted_as_full_pairs(self):
        delta, count = report.paired_delta(self.data['volume'], 'Qwen/Qwen3-4B', 'gen')
        self.assertEqual(count, 1)
        self.assertAlmostEqual(delta * 100, -0.94, places=7)
        qwen = report.by_seed(self.data['volume'], 'Qwen/Qwen2.5-7B-Instruct', 'clean', 'gen')
        self.assertEqual(set(qwen), {72, 73})
        summary = json.loads(self.files['data/report_checks.json'])
        self.assertEqual(summary['clean_bmk_full_pair_models'], 5)
        self.assertEqual(summary['clean_bmk_wins'], 4)
        self.assertEqual(summary['nll_wins'], 1)
        self.assertEqual(summary['fixed_records'], 93)
        self.assertEqual(summary['knowledge_bound_records'], 93)

    def test_reference_relative_objective_requires_same_seed_clean(self):
        rows = self.data['volume']
        report.add_objective(rows)
        qwen = report.by_seed(rows, 'Qwen/Qwen2.5-7B-Instruct', 'selected_recipe', 'repair_rel_nll')
        self.assertEqual(set(qwen), {72, 73})
        for r in rows:
            if r['arm'] == 'clean':
                self.assertEqual(r['repair_rel_nll'], 0)

    def test_forecast_baselines_use_same_cells(self):
        rows = list(csv.DictReader(io.StringIO(report.read_input('forecast_errors.csv'))))
        keys = ['model', 'repair_axis', 'endpoint', 'metric', 'target_q']
        for metric in ('mean_per_item_nll', 'bits_per_byte'):
            sets = {}
            values = {}
            for method in ('fixed_tau_saturating', 'nearest_q10', 'log_linear', 'raw_linear'):
                rs = [r for r in rows if r['metric'] == metric and r['predictor'] == method
                      and r['primary_comparison_eligible'] == 'True' and r['absolute_error']]
                sets[method] = {tuple(r[k] for k in keys) for r in rs}
                self.assertEqual(len(sets[method]), len(rs))
                values[method] = statistics.mean(float(r['absolute_error']) for r in rs)
            self.assertEqual(sets['fixed_tau_saturating'], sets['nearest_q10'])
            self.assertEqual(len(sets['nearest_q10']), 300)
            self.assertEqual(len(set.intersection(*sets.values())), 297)
            self.assertGreater(values['fixed_tau_saturating'], values['nearest_q10'])

    def test_calibration_budget_comparison_has_common_cells(self):
        rows = list(csv.DictReader(io.StringIO(report.read_input('calibration_budget_errors.csv'))))
        rules = {r['rule'] for r in rows}
        sets = [{(r['model'], r['repair_axis'], r['endpoint'], r['metric'], r['dose'])
                 for r in rows if r['rule'] == rule and r['relative_absolute_error']}
                for rule in sorted(rules)]
        self.assertTrue(all(s == sets[0] for s in sets))

    def test_historical_selected_equals_uniform_and_preserves_scale(self):
        rows = json.loads(report.read_input('historical.json'))['paired']
        selected = [r for r in rows if 'MODEL_SELECTED' in r['recipe_role'] and int(r['seed']) in (72, 73)]
        self.assertEqual(len(selected), 12)
        self.assertEqual({int(r['N']) for r in selected}, {1920, 3968, 9984})
        for r in selected:
            shares = [float(r[g]) for g in report.GROUPS]
            expected = [0, .25, .25, .25, .25] if r['model'] == 'Qwen/Qwen3-8B' else [.2] * 5
            for a, b in zip(shares, expected):
                self.assertAlmostEqual(a, b)

    def test_input_hashes(self):
        manifest = json.loads((report.HERE / 'data/input_manifest.json').read_text())
        for record in manifest['inputs']:
            content = report.read_input(record['logical_name']).encode()
            self.assertEqual(hashlib.sha256(content).hexdigest(), record['decoded_sha256'])
            stored = (report.HERE / record['stored_path']).read_bytes()
            self.assertEqual(hashlib.sha256(stored).hexdigest(), record['stored_sha256'])

    def test_committed_artifacts_reproduce_exactly(self):
        for name, content in self.files.items():
            self.assertEqual((report.HERE / name).read_text(), content)

    def test_markdown_navigation_and_details(self):
        for name, content in self.files.items():
            if not name.endswith('.md'):
                continue
            self.assertEqual(content.count('<details>'), content.count('</details>'))
            for target in re.findall(r'\]\(([^)]+)\)', content):
                if '://' not in target:
                    self.assertTrue((report.HERE / target.split('#')[0]).is_file(), target)


if __name__ == '__main__':
    unittest.main()
