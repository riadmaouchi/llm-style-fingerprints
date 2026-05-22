"""Tests pour src/stats.py."""

import numpy as np
import pytest

from src.stats import bootstrap_ci, intra_variance, pairwise_tests, permutation_test

RNG = np.random.default_rng(0)
GROUP_A = list(RNG.normal(0.20, 0.05, 16))   # GPT-4-like shifts
GROUP_B = list(RNG.normal(0.10, 0.05, 16))   # Mistral-like shifts (clairement différent)
GROUP_C = list(RNG.normal(0.21, 0.05, 16))   # Quasi-identique à GROUP_A


class TestBootstrapCI:
    def test_returns_tuple_of_two_floats(self):
        lo, hi = bootstrap_ci(GROUP_A)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lo_less_than_hi(self):
        lo, hi = bootstrap_ci(GROUP_A)
        assert lo < hi

    def test_mean_inside_ci(self):
        lo, hi = bootstrap_ci(GROUP_A, n_boot=2000)
        assert lo <= np.mean(GROUP_A) <= hi

    def test_ci_95_wider_than_ci_50(self):
        lo_50, hi_50 = bootstrap_ci(GROUP_A, ci=0.50, n_boot=2000)
        lo_95, hi_95 = bootstrap_ci(GROUP_A, ci=0.95, n_boot=2000)
        assert (hi_95 - lo_95) > (hi_50 - lo_50)

    def test_reproducible_with_same_seed(self):
        r1 = bootstrap_ci(GROUP_A, rng=np.random.default_rng(42))
        r2 = bootstrap_ci(GROUP_A, rng=np.random.default_rng(42))
        assert r1 == r2

    def test_vectorized_bootstrap_matches_loop(self):
        """La version vectorisée doit produire des résultats équivalents."""
        lo, hi = bootstrap_ci(GROUP_A, n_boot=500, rng=np.random.default_rng(7))
        assert 0.0 <= lo <= hi <= 1.0


class TestPermutationTest:
    def test_returns_required_keys(self):
        result = permutation_test(GROUP_A, GROUP_B, n_perm=200)
        for key in ("observed", "p_value", "null_distribution", "significant_05", "significant_01"):
            assert key in result

    def test_clearly_different_groups_are_significant(self):
        result = permutation_test(GROUP_A, GROUP_B, n_perm=2000,
                                  rng=np.random.default_rng(42))
        assert result["significant_05"], f"p={result['p_value']}"

    def test_identical_groups_not_significant(self):
        result = permutation_test(GROUP_A, GROUP_A, n_perm=2000,
                                  rng=np.random.default_rng(42))
        assert not result["significant_05"]

    def test_p_value_lower_bound(self):
        result = permutation_test(GROUP_A, GROUP_B, n_perm=100)
        assert result["p_value"] >= 1 / 100

    def test_observed_equals_abs_diff_means(self):
        result = permutation_test(GROUP_A, GROUP_B, n_perm=100)
        expected = abs(np.mean(GROUP_A) - np.mean(GROUP_B))
        assert abs(result["observed"] - expected) < 1e-10

    def test_null_distribution_shape(self):
        n_perm = 300
        result = permutation_test(GROUP_A, GROUP_B, n_perm=n_perm)
        assert len(result["null_distribution"]) == n_perm


class TestPairwiseTests:
    SHIFTS = {"GPT-4": GROUP_A, "Mistral": GROUP_B, "Clone": GROUP_C}

    def test_number_of_pairs(self):
        results = pairwise_tests(self.SHIFTS)
        assert len(results) == 3  # C(3,2)

    def test_sorted_by_p_corrected(self):
        results = pairwise_tests(self.SHIFTS)
        ps = [r["p_corrected"] for r in results]
        assert ps == sorted(ps)

    def test_known_significant_pair(self):
        results = pairwise_tests(self.SHIFTS, correction="bonferroni")
        gpt_vs_mistral = next(
            r for r in results if {"r" for r in [r["model_a"], r["model_b"]]}
            == {"GPT-4", "Mistral"} or
            (r["model_a"] in ("GPT-4", "Mistral") and r["model_b"] in ("GPT-4", "Mistral"))
        )
        assert gpt_vs_mistral["significant_05"]

    def test_invalid_correction_raises(self):
        with pytest.raises(ValueError, match="correction must be one of"):
            pairwise_tests(self.SHIFTS, correction="fdr_bh_invalid")

    def test_holm_correction_accepted(self):
        results = pairwise_tests(self.SHIFTS, correction="holm")
        assert all("p_corrected" in r for r in results)

    def test_no_correction_accepted(self):
        results = pairwise_tests(self.SHIFTS, correction="none")
        for r in results:
            assert r["p_corrected"] == r["p_raw"]


_T1 = ("il marchait dans la nuit sous la pluie et le vent glacial "
       "et il ne voyait plus rien devant lui dans l'obscurité profonde")
_T2 = ("elle regardait la mer depuis la fenêtre et pensait à lui "
       "depuis que le soleil s'était couché derrière les collines lointaines")
_T3 = ("le vieux chêne se dressait au milieu du champ et ses branches "
       "s'étiraient vers le ciel comme des bras tendus vers la lumière")


class TestIntraVariance:
    def test_returns_dict(self):
        from src.stylometry import StyleAnalyzer
        sa     = StyleAnalyzer()
        result = intra_variance({"groupe": [_T1, _T2, _T3, _T1, _T2, _T3]}, sa)
        assert "groupe" in result
        assert 0.0 <= result["groupe"] <= 1.0

    def test_single_text_returns_zero(self):
        from src.stylometry import StyleAnalyzer
        sa = StyleAnalyzer()
        result = intra_variance({"solo": [_T1]}, sa)
        assert result["solo"] == 0.0

    def test_identical_texts_have_zero_variance(self):
        from src.stylometry import StyleAnalyzer
        sa   = StyleAnalyzer()
        result = intra_variance({"même": [_T1, _T1, _T1]}, sa)
        assert result["même"] == pytest.approx(0.0, abs=1e-9)
