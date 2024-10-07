import numpy as np
from psruq.metrics.create_specific_risks import (
    check_scalar_product,
    get_energy_inner,
    get_energy_outer,
    get_risk_approximation,
)
from psruq.metrics.constants import ApproximationType, GName, RiskType

if __name__ == "__main__":
    N_members, N_objects, N_classes = 100, 5, 10  # Example dimensions
    # N_members, N_objects, N_classes = 2, 3, 4  # Example dimensions

    max_concentration = 10
    probs = np.random.dirichlet(
        alpha=np.random.rand(N_classes) * max_concentration, size=(N_members, N_objects)
    ).reshape(N_members, N_objects, N_classes)
    logits = np.log(probs)
    T = 1.0

    results = {}

    for g_name in [el for el in GName]:
        for risk_type in [el for el in RiskType]:
            for gt_approx in [el for el in ApproximationType]:
                for pred_approx in [el for el in ApproximationType]:
                    res = get_risk_approximation(
                        g_name=g_name,
                        risk_type=risk_type,
                        logits=logits,
                        gt_approx=gt_approx,
                        pred_approx=pred_approx,
                        probabilities=probs,
                        T=T,
                    )

                    if risk_type == RiskType.BAYES_RISK:
                        results[
                            f"{g_name.value} {risk_type.value} {gt_approx.value}"
                        ] = res
                    else:
                        results[
                            f"{g_name.value} {risk_type.value} {gt_approx.value} {pred_approx.value}"
                        ] = res

        results[f"{g_name.value} scalar product"] = check_scalar_product(
            g_name=g_name, logits=logits, T=T, probabilities=probs
        )

    # print(list(results.keys()))

    for g_name in [el.value for el in GName]:
        R_tot_1_1 = results[f"{g_name} TotalRisk outer outer"]
        R_tot_2_1 = results[f"{g_name} TotalRisk inner outer"]
        R_tot_1_2 = results[f"{g_name} TotalRisk outer inner"]
        R_tot_2_2 = results[f"{g_name} TotalRisk inner inner"]
        R_tot_3_1 = results[f"{g_name} TotalRisk central outer"]
        R_tot_1_3 = results[f"{g_name} TotalRisk outer central"]
        R_tot_3_2 = results[f"{g_name} TotalRisk central inner"]
        R_tot_2_3 = results[f"{g_name} TotalRisk inner central"]
        R_tot_3_3 = results[f"{g_name} TotalRisk central central"]

        R_exc_1_1 = results[f"{g_name} ExcessRisk outer outer"]
        R_exc_2_1 = results[f"{g_name} ExcessRisk inner outer"]
        R_exc_1_2 = results[f"{g_name} ExcessRisk outer inner"]
        R_exc_2_2 = results[f"{g_name} ExcessRisk inner inner"]
        R_exc_3_1 = results[f"{g_name} ExcessRisk central outer"]
        R_exc_1_3 = results[f"{g_name} ExcessRisk outer central"]
        R_exc_3_2 = results[f"{g_name} ExcessRisk central inner"]
        R_exc_2_3 = results[f"{g_name} ExcessRisk inner central"]
        R_exc_3_3 = results[f"{g_name} ExcessRisk central central"]

        R_bay_1 = results[f"{g_name} BayesRisk outer"]
        R_bay_2 = results[f"{g_name} BayesRisk inner"]
        R_bay_3 = results[f"{g_name} BayesRisk central"]

        scalar_product = results[f"{g_name} scalar product"]

        assert np.all(
            R_exc_2_2 == np.zeros_like(R_exc_2_2)
        ), f"{g_name}: R_exc_2_2 should be 0"

        assert np.all(
            R_exc_3_3 == np.zeros_like(R_exc_3_3)
        ), f"{g_name}: R_exc_3_3 should be 0"

        assert np.all(R_exc_1_1 >= R_exc_2_1), f"{g_name}: R_exc_1_1 >= R_exc_2_1"
        assert np.all(R_exc_1_1 >= R_exc_1_2), f"{g_name}: R_exc_1_1 >= R_exc_1_2"
        assert np.all(
            np.isclose(R_exc_1_1 - R_exc_2_1, R_exc_1_2)
        ), f"{g_name}: R_exc_1_1 - R_exc_2_1 = R_exc_1_2"

        assert np.all(
            np.isclose(R_exc_2_1, R_exc_1_1 - R_exc_1_2)
        ), f"{g_name}: R_exc_2_1 = R_exc_1_1 - R_exc_1_2"

        assert np.all(
            np.all(np.isclose(R_bay_2 - R_bay_1, R_exc_1_2))
        ), f"{g_name}: R_bay_2 - R_bay_1 = R_exc_1_2"

        assert np.all(
            np.isclose(R_tot_1_1, R_tot_2_1)
        ), f"{g_name}: R_tot_1_1 = R_tot_2_1"

        assert np.all(
            np.isclose(R_tot_1_1 - R_tot_1_2, R_tot_2_1 - R_tot_2_2)
        ), f"{g_name}: R_tot_1_1 - R_tot_1_2 = R_tot_2_1 - R_tot_2_2"

        assert np.all(
            np.isclose(R_tot_1_1 - R_tot_1_2, R_exc_1_1 - R_exc_1_2)
        ), f"{g_name}: R_tot_1_1 - R_tot_1_2 = R_exc_1_1 - R_exc_1_2 = R_exc_2_1"

        assert np.all(np.isclose(R_exc_1_3 + R_exc_3_1, R_exc_1_2 + R_exc_2_1)), (
            f"{g_name}: R_exc_1_3 + R_exc_3_1 = R_exc_1_2 + R_exc_2_1,"
            f"{np.max(np.abs(R_exc_1_3 + R_exc_3_1 - (R_exc_1_2 + R_exc_2_1)))}"
        )

        assert np.all(
            np.isclose(R_exc_1_1, R_exc_1_3 + R_exc_3_1)
        ), f"{g_name}: R_exc_1_1 = R_exc_1_3 + R_exc_3_1, {np.max(np.abs(R_exc_1_1 - (R_exc_1_3 + R_exc_3_1)))}"

        assert np.all(
            np.isclose(R_exc_1_1, R_exc_1_2 + R_exc_2_1)
        ), f"{g_name}: R_exc_1_1 = R_exc_1_2 + R_exc_2_1, {np.max(np.abs(R_exc_1_1 - (R_exc_1_2 + R_exc_2_1)))}"

        assert np.all(
            np.isclose(scalar_product, 0.0)
        ), f"{g_name}: max abs scalar product is {np.max(np.abs(scalar_product))}"

        if g_name == GName.LOG_SCORE.value:
            energy_diff = get_energy_inner(logits=logits, T=T) - get_energy_outer(
                logits=logits, T=T
            )
            assert np.all(
                np.isclose(
                    R_exc_3_1,
                    np.squeeze(energy_diff) / T,
                )
            ), f"{g_name}: Energy does not coincide!"
