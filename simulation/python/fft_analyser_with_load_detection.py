"""
LTspice FFT analyser with harmonic-based load detection.

This script keeps the original FFT/THD workflow, and adds a load
identification stage based on harmonic signatures.

Example signatures:
    LOAD 1 -> H3 = 20% of fundamental, H7 = 14% of fundamental
    LOAD 2 -> H5 = 20% of fundamental, H11 = 14% of fundamental

The detector compares the measured harmonic percentages against the
reference signatures and reports which load(s) are a good match.

Important:
    The reference harmonic percentages should ideally be obtained at the
    same PCC measurement point, fundamental frequency, and operating
    conditions as the signal being analysed.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# LTspice exported data file.
# Change this path to the actual location of your LTspice export.
DATA_FILE = "data/test2.txt"

# Column names used by the LTspice export.
TIME_COLUMN = "time"
SIGNAL_COLUMN = "V(N001,N002)"

# Fundamental frequency of the system.
FUNDAMENTAL_FREQUENCY = 50.0  # Hz

# Sampling frequency used after interpolation.
# The original code used 50 kHz, so this is intentionally retained.
SAMPLING_FREQUENCY = 50_000.0  # Hz

# Number of harmonic orders to extract.
MAX_HARMONIC_ORDER = 20


# -----------------------------------------------------------------------------
# LOAD HARMONIC SIGNATURE DATABASE
# -----------------------------------------------------------------------------
# Each value is the expected harmonic amplitude as a percentage of the
# fundamental amplitude.
#
# Example:
#     3: 20.0
#
# means:
#     H3 is expected to be approximately 20% of H1.
#
# Add/remove loads here as the project develops. The detection algorithm
# below does not need to be changed when new load signatures are added.
# -----------------------------------------------------------------------------
LOAD_SIGNATURES = {
    "LOAD 1": {
        3: 5.0,
        7: 5.0,
    },

    "LOAD 2": {
        5: 5.0,
        11: 5.0,
    },
}


# -----------------------------------------------------------------------------
# DETECTION SETTINGS
# -----------------------------------------------------------------------------

# Allowed relative error between a measured harmonic percentage and the
# reference percentage.
#
# Example:
#     expected H5 = 20%
#     tolerance = 0.40
#
# gives an acceptable range of:
#     12% <= H5 <= 28%
#
# This should be tuned using experimental/simulation data.
REFERENCE_TOLERANCE = 0.001


# A harmonic below this percentage of H1 is treated as effectively absent.
# This prevents numerical noise or tiny FFT components from being considered
# a detected harmonic.
MIN_HARMONIC_PERCENT = 2.0


# Number of reference harmonics that must match before a load is declared
# detected.
#
# For the example loads, each load has two characteristic harmonics, so
# requiring both is a conservative choice.
MINIMUM_MATCHES = 2


# Overall score required to classify a load.
#
# A score of 1.0 means the measured harmonic percentages exactly match the
# reference values. A score of 0 means none of the reference harmonics match.
MINIMUM_DETECTION_SCORE = 0.55


# =============================================================================
# DATA READING
# =============================================================================

def load_simulation_data(file_path: str) -> pd.DataFrame:
    """
    Read the LTspice ASCII export into a pandas DataFrame.

    The original script uses whitespace-separated LTspice output, so the
    same parsing approach is retained.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find LTspice data file: {path}"
        )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
    )

    required_columns = {TIME_COLUMN, SIGNAL_COLUMN}

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The LTspice export is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    return df


# =============================================================================
# SIGNAL PRE-PROCESSING
# =============================================================================

def resample_signal(
    time: np.ndarray,
    signal: np.ndarray,
    sampling_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate the non-uniform LTspice transient data onto a uniform
    time grid.

    LTspice transient exports can contain non-uniform time steps. FFT
    algorithms assume uniformly spaced samples, therefore interpolation
    is performed before the FFT.

    This preserves the original script's interpolation approach.
    """

    dt = 1.0 / sampling_frequency

    # Generate a uniform time axis covering the original simulation interval.
    t_uniform = np.arange(
        time[0],
        time[-1],
        dt,
    )

    # Linear interpolation is the same method used in the original script.
    signal_uniform = np.interp(
        t_uniform,
        time,
        signal,
    )

    return t_uniform, signal_uniform


# =============================================================================
# FFT CALCULATION
# =============================================================================

def calculate_fft(
    signal: np.ndarray,
    sampling_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the one-sided FFT magnitude spectrum.

    The original code uses rfft(), which is appropriate for a real-valued
    LTspice waveform. The same magnitude scaling is retained.
    """

    # Remove DC offset before calculating the FFT.
    signal = signal - np.mean(signal)

    number_of_samples = len(signal)

    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / sampling_frequency,
    )

    # One-sided amplitude spectrum.
    magnitude = np.abs(fft_values) * 2.0 / number_of_samples

    return frequencies, magnitude


# =============================================================================
# HARMONIC EXTRACTION
# =============================================================================

def extract_harmonics(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    fundamental_frequency: float,
    max_harmonic_order: int,
) -> pd.DataFrame:
    """
    Extract the FFT magnitude at H1, H2, ..., HN.

    The original implementation selected the FFT bin closest to each
    theoretical harmonic frequency. That method is retained here.

    Returns a DataFrame because it makes the harmonic data easier to inspect,
    print, save, and use in later analysis.
    """

    orders = np.arange(
        1,
        max_harmonic_order + 1,
    )

    harmonic_rows = []

    for order in orders:

        target_frequency = order * fundamental_frequency

        # Find the FFT bin nearest to the theoretical harmonic frequency.
        index = np.argmin(
            np.abs(frequencies - target_frequency)
        )

        harmonic_rows.append(
            {
                "order": int(order),
                "frequency_hz": target_frequency,
                "magnitude": magnitude[index],
                "fft_frequency_hz": frequencies[index],
            }
        )

    harmonics_df = pd.DataFrame(harmonic_rows)

    # Convert each harmonic magnitude to a percentage of the fundamental.
    fundamental_magnitude = harmonics_df.loc[
        harmonics_df["order"] == 1,
        "magnitude",
    ].iloc[0]

    if np.isclose(fundamental_magnitude, 0.0):
        raise ValueError(
            "The fundamental amplitude is approximately zero. "
            "Harmonic percentages and load detection cannot be calculated."
        )

    harmonics_df["percent_of_fundamental"] = (
        harmonics_df["magnitude"]
        / fundamental_magnitude
        * 100.0
    )

    return harmonics_df


# =============================================================================
# LOAD DETECTION
# =============================================================================

def calculate_load_match(
    harmonics_df: pd.DataFrame,
    reference_signature: dict[int, float],
    tolerance: float,
) -> dict:
    """
    Compare one load's harmonic signature against the measured FFT.

    Detection is based on the characteristic harmonic group of the load.

    For every reference harmonic:

        measured_percentage
                  |
                  v
        compare with expected_percentage
                  |
                  v
        calculate relative error
                  |
                  v
        convert error into a 0..1 match score

    A harmonic outside the tolerance is considered a non-match.

    The final load score is the average of the individual harmonic scores.
    """

    measured_by_order = (
        harmonics_df
        .set_index("order")["percent_of_fundamental"]
        .to_dict()
    )

    harmonic_results = []
    scores = []
    matched_harmonics = []

    for harmonic_order, expected_percentage in reference_signature.items():

        measured_percentage = measured_by_order.get(
            harmonic_order,
            0.0,
        )

        # If the measured harmonic is below the noise/visibility threshold,
        # treat it as absent.
        if measured_percentage < MIN_HARMONIC_PERCENT:

            harmonic_results.append(
                {
                    "order": harmonic_order,
                    "expected_percent": expected_percentage,
                    "measured_percent": measured_percentage,
                    "relative_error": np.inf,
                    "match_score": 0.0,
                    "matched": False,
                }
            )

            scores.append(0.0)
            continue

        # Relative error is normalized against the reference amplitude.
        relative_error = abs(
            measured_percentage - expected_percentage
        ) / expected_percentage

        # Convert the error into a score.
        #
        # Exact match:
        #     error = 0       -> score = 1
        #
        # At tolerance limit:
        #     error = tolerance -> score = 0
        #
        # Beyond tolerance:
        #     score is clipped to 0.
        match_score = max(
            0.0,
            1.0 - relative_error / tolerance,
        )

        matched = relative_error <= tolerance

        if matched:
            matched_harmonics.append(harmonic_order)

        harmonic_results.append(
            {
                "order": harmonic_order,
                "expected_percent": expected_percentage,
                "measured_percent": measured_percentage,
                "relative_error": relative_error,
                "match_score": match_score,
                "matched": matched,
            }
        )

        scores.append(match_score)

    overall_score = float(np.mean(scores)) if scores else 0.0

    return {
        "score": overall_score,
        "matched_harmonics": matched_harmonics,
        "number_of_matches": len(matched_harmonics),
        "number_of_reference_harmonics": len(reference_signature),
        "details": harmonic_results,
    }


def detect_loads(
    harmonics_df: pd.DataFrame,
    load_signatures: dict[str, dict[int, float]],
    tolerance: float,
    minimum_matches: int,
    minimum_score: float,
) -> list[dict]:
    """
    Run the harmonic detector against every load in LOAD_SIGNATURES.

    Important:
        Loads are evaluated independently.

    Therefore, if the PCC contains a combination of LOAD 1 and LOAD 2,
    the detector can report both rather than forcing the signal to belong
    to only one load.

    The returned list is sorted from the strongest match to the weakest.
    """

    detection_results = []

    for load_name, signature in load_signatures.items():

        result = calculate_load_match(
            harmonics_df=harmonics_df,
            reference_signature=signature,
            tolerance=tolerance,
        )

        detected = (
            result["number_of_matches"] >= minimum_matches
            and result["score"] >= minimum_score
        )

        result["load_name"] = load_name
        result["detected"] = detected

        detection_results.append(result)

    # Highest-confidence matches first.
    detection_results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return detection_results


# =============================================================================
# PRINTING / REPORTING
# =============================================================================

def print_harmonic_breakdown(harmonics_df: pd.DataFrame) -> None:
    """Print the harmonic breakdown in a readable format."""

    print("\nHARMONIC BREAKDOWN")
    print("-" * 72)

    for _, row in harmonics_df.iterrows():

        print(
            f"H{int(row['order']):2d}  "
            f"{row['frequency_hz']:7.1f} Hz  "
            f"Amplitude = {row['magnitude']:.4f}  "
            f"({row['percent_of_fundamental']:.2f}% of H1)"
        )


def print_load_detection_report(
    detection_results: list[dict],
) -> None:
    """
    Print the load identification result.

    This is the part that generates the requested interpretation, e.g.:

        The FFT of the PCC has been analysed and the harmonics seem
        to come from LOAD 2.
    """

    print("\nLOAD DETECTION")
    print("-" * 72)

    detected_loads = [
        result
        for result in detection_results
        if result["detected"]
    ]

    if not detected_loads:

        print(
            "No configured load signature was detected "
            "with the current thresholds."
        )

        print(
            "Consider adjusting REFERENCE_TOLERANCE, "
            "MIN_HARMONIC_PERCENT, or the reference signatures."
        )

        return

    if len(detected_loads) == 1:

        load_name = detected_loads[0]["load_name"]

        print(
            f"The FFT of the PCC has been analysed and the harmonics "
            f"seem to come from {load_name}."
        )

    else:

        load_names = [
            result["load_name"]
            for result in detected_loads
        ]

        print(
            "The FFT of the PCC has been analysed and the harmonics "
            "seem to contain contributions from: "
            + ", ".join(load_names)
            + "."
        )

    print("\nDetection details:")

    for result in detection_results:

        print(
            f"\n{result['load_name']}: "
            f"score = {result['score']:.2f}, "
            f"matched = "
            f"{result['number_of_matches']}/"
            f"{result['number_of_reference_harmonics']}, "
            f"detected = {result['detected']}"
        )

        for detail in result["details"]:

            error_text = (
                "n/a"
                if not np.isfinite(detail["relative_error"])
                else f"{detail['relative_error'] * 100:.1f}%"
            )

            print(
                f"    H{detail['order']:2d}: "
                f"measured = {detail['measured_percent']:7.2f}%  "
                f"expected = {detail['expected_percent']:7.2f}%  "
                f"error = {error_text:>8}  "
                f"match = {detail['matched']}"
            )


# =============================================================================
# PLOTTING
# =============================================================================

def plot_fft(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
) -> None:
    """
    Plot the complete FFT spectrum.

    The original plot limits and labels are retained.
    """

    plt.figure(figsize=(12, 5))

    plt.plot(
        frequencies,
        magnitude,
    )

    plt.xlim(0, 500)
    plt.xticks(np.arange(0, 501, 50))

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.ylim(0, 800)
    plt.yticks(np.arange(0, 801, 50))

    plt.grid(True)

    #plt.show()


def plot_harmonics(harmonics_df: pd.DataFrame) -> None:
    """Plot the harmonic magnitude breakdown."""

    orders = harmonics_df["order"].to_numpy()
    magnitudes = harmonics_df["magnitude"].to_numpy()

    plt.figure(figsize=(10, 4))

    plt.bar(
        orders,
        magnitudes,
    )

    plt.xlabel("Harmonic Order")
    plt.xlim(0, 15)
    plt.xticks(np.arange(0, 16, 1))

    plt.ylabel("Magnitude")

    plt.grid(True)

    plt.show()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main() -> None:
    """
    Run the complete LTspice FFT analysis.

    The order of operations remains essentially the same as the original:

        1. Read LTspice data
        2. Inspect original time step
        3. Interpolate to 50 kHz
        4. Remove DC
        5. Calculate FFT
        6. Plot FFT
        7. Extract H1..H20
        8. Print harmonic breakdown
        9. Plot harmonic breakdown
       10. Calculate THD
       11. Detect load from harmonic signature
    """

    # -------------------------------------------------------------------------
    # 1. READ LTspice DATA
    # -------------------------------------------------------------------------

    df = load_simulation_data(DATA_FILE)

    # Keep the equivalent of the original df.head() available when running
    # interactively. Uncomment the next line if desired.
    # print(df.head())

    time = df[TIME_COLUMN].to_numpy(dtype=float)
    signal = df[SIGNAL_COLUMN].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # 2. INSPECT LTspice TIME STEP
    # -------------------------------------------------------------------------

    original_dt = np.diff(time)

    print(f"Original minimum dt = {original_dt.min():.6e} s")
    print(f"Original maximum dt = {original_dt.max():.6e} s")

    # -------------------------------------------------------------------------
    # 3. INTERPOLATE TO UNIFORM SAMPLING RATE
    # -------------------------------------------------------------------------

    time, signal = resample_signal(
        time=time,
        signal=signal,
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    # -------------------------------------------------------------------------
    # 4. FFT
    # -------------------------------------------------------------------------

    frequencies, magnitude = calculate_fft(
        signal=signal,
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    # -------------------------------------------------------------------------
    # 5. COMPLETE FFT PLOT
    # -------------------------------------------------------------------------

    plot_fft(
        frequencies=frequencies,
        magnitude=magnitude,
    )

    # -------------------------------------------------------------------------
    # 6. EXTRACT HARMONICS
    # -------------------------------------------------------------------------

    harmonics_df = extract_harmonics(
        frequencies=frequencies,
        magnitude=magnitude,
        fundamental_frequency=FUNDAMENTAL_FREQUENCY,
        max_harmonic_order=MAX_HARMONIC_ORDER,
    )

    # -------------------------------------------------------------------------
    # 7. PRINT HARMONIC BREAKDOWN
    # -------------------------------------------------------------------------

    print_harmonic_breakdown(harmonics_df)

    # -------------------------------------------------------------------------
    # 8. HARMONIC BAR PLOT
    # -------------------------------------------------------------------------

    plot_harmonics(harmonics_df)

    # -------------------------------------------------------------------------
    # 9. THD
    # -------------------------------------------------------------------------

    fundamental_amplitude = harmonics_df.loc[
        harmonics_df["order"] == 1,
        "magnitude",
    ].iloc[0]

    harmonic_amplitudes = harmonics_df.loc[
        harmonics_df["order"] >= 2,
        "magnitude",
    ].to_numpy()

    THD = (
        np.sqrt(np.sum(harmonic_amplitudes ** 2))
        / fundamental_amplitude
        * 100.0
    )

    print(f"\nTHD = {THD:.2f}%")

    # -------------------------------------------------------------------------
    # 10. LOAD DETECTION
    # -------------------------------------------------------------------------

    detection_results = detect_loads(
        harmonics_df=harmonics_df,
        load_signatures=LOAD_SIGNATURES,
        tolerance=REFERENCE_TOLERANCE,
        minimum_matches=MINIMUM_MATCHES,
        minimum_score=MINIMUM_DETECTION_SCORE,
    )

    print_load_detection_report(detection_results)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
