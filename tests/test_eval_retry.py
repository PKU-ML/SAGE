import unittest
from sage.runtime.eval_resume import retry_failed_tail, validate_resume


class RetryTest(unittest.TestCase):
    def test_old_stateless_gaussian_preserves_error_and_good_rows(self):
        protocol = {'controller': 'gaussian_lewm'}
        previous = {'protocol': protocol, 'proposal_rng_state': [1, 2],
            'results': [{'query_id': 'a'}, {'query_id': 'b', 'error': 'out of memory'}]}
        actual = retry_failed_tail(previous, protocol, ['a', 'b'], [1, 2])
        self.assertEqual(validate_resume(actual, protocol, ['a', 'b']), [1, 2])
        self.assertEqual(len(actual['results']), 1)
        self.assertEqual(actual['failed_attempts'][0]['query_id'], 'b')
        self.assertEqual(len(previous['results']), 2)

    def test_unknown_prior_rng_cannot_be_guessed(self):
        protocol = {'controller': 'sage'}
        previous = {'protocol': protocol, 'proposal_rng_state': [1],
            'results': [{'query_id': 'a', 'error': 'oom'}]}
        with self.assertRaisesRegex(ValueError, 'pre-query RNG'):
            retry_failed_tail(previous, protocol, ['a'], [1])
        previous['results'][0]['proposal_rng_state_before'] = [2]
        actual = retry_failed_tail(previous, protocol, ['a'], [1])
        self.assertEqual(actual['proposal_rng_state'], [2])


if __name__ == '__main__':
    unittest.main()
