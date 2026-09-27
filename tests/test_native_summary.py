import copy
import unittest

from sage.summarize_native import validate_group


class NativeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.queries = [{'query_id': str(i)} for i in range(50)]
        self.payload = {'protocol': {'controller': 'sage', 'seed': 43,
            'query_start': 50, 'num_queries': 50, 'query_manifest_sha256': 'abc',
            'chunk_raw_actions': 15, 'recovery_raw_actions': 120, 'cem_rounds': 1,
            'num_candidates': 64}, 'results': [dict(q, official_success=i < 30)
            for i, q in enumerate(self.queries)]}

    def validate(self, payload):
        return validate_group(payload, self.queries, robotwin=True, method='sage',
                              group=1, manifest_sha='abc')

    def test_count(self):
        self.assertEqual(self.validate(self.payload), 30)

    def test_scene2_historical_seed(self):
        queries = [{'query_index': i} for i in range(50)]
        payload = {'protocol': {'suite': 'libero_scene2', 'method': 'sage',
            'seed': 20260829, 'query_start': 0, 'num_queries': 50,
            'commitment': 15, 'recovery': 120, 'rank_rounds': 1, 'K': 64},
            'query_manifest_sha256': 'abc',
            'results': [dict(q, official_success_ever=True) for q in queries]}
        def check():
            return validate_group(payload, queries, robotwin=False, method='sage',
                group=0, manifest_sha='abc', suite='libero_scene2')
        self.assertEqual(check(), 50)
        payload['protocol']['seed'] = 42
        with self.assertRaisesRegex(ValueError, 'sampling seed'):
            check()

    def test_incomplete_is_not_a_success_rate(self):
        self.payload['results'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.validate(self.payload)

    def test_error_is_not_a_task_failure(self):
        self.payload['results'][0]['error'] = 'bad runtime'
        with self.assertRaisesRegex(ValueError, 'Runtime errors'):
            self.validate(self.payload)

    def test_seed_and_duplicates(self):
        for mutation in ('seed', 'duplicate'):
            payload = copy.deepcopy(self.payload)
            if mutation == 'seed':
                payload['protocol']['seed'] = 42
            else:
                payload['results'][1] = payload['results'][0]
            with self.assertRaises(ValueError):
                self.validate(payload)


if __name__ == '__main__':
    unittest.main()
